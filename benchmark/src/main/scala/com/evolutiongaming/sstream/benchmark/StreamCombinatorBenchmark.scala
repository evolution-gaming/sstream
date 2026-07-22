package com.evolutiongaming.sstream.benchmark

import cats.effect.{IO, SyncIO}
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*
import com.evolutiongaming.sstream.Stream
import com.evolutiongaming.sstream.Stream.StreamOps
import org.openjdk.jmh.annotations.*

import java.util.concurrent.TimeUnit

/*
 * gc.alloc.rate.norm (B/op), size=10000 — baseline -> stateful/statefulM/flatMapLast var refactor:
 *
 *   dropEvery fanout | SyncIO                          | IO
 *   0         1      | 11,679,196 ->  9,804,203 (-16%) | 10,154,934 ->  8,654,951 (-15%)
 *   0         4      | 30,803,994 -> 23,978,982 (-22%) | 25,904,809 -> 19,904,795 (-23%)
 *   1         1      | 11,679,196 ->  9,804,203 (-16%) | 10,154,931 ->  8,654,932 (-15%)
 *   1         4      | 22,554,055 -> 18,804,069 (-17%) | 19,154,865 -> 16,154,824 (-16%)
 */
@State(Scope.Benchmark)
@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
@Fork(1)
@Warmup(iterations = 3, time = 3, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
class StreamCombinatorBenchmark {

  import StreamCombinatorBenchmark.*

  @Param(Array("10000"))
  var size: Int = _

  // events per record: 1 == kafka-journal single-event append; 4 == batched append
  @Param(Array("1", "4"))
  var fanout: Int = _

  // fraction of events dropped by the dedup filter: 0 == kept-heavy tail; 1 == duplicated tail
  @Param(Array("0", "1"))
  var dropEvery: Int = _

  private var records: List[Int] = _

  @Setup(Level.Trial)
  def setup(): Unit =
    records = (1 to size).toList

  @Benchmark
  def pipeline(): Long = {
    val fo = fanout
    val drop = dropEvery
    val source: Stream[SyncIO, Int] = Stream[SyncIO].apply(records)

    val stream = source
      .flatMap { r =>                                        // record dispatch (single/empty)
        if (r % 16 == 0) Stream[SyncIO].empty[Int] else Stream[SyncIO].single(r)
      }
      .filter { _ >= 0 }                                     // range guard
      .flatMap { r => Stream.lift(SyncIO(r)) }               // effectful decode (F -> Stream)
      .flatMap { r => Stream[SyncIO].apply(expand(r, fo)) }  // event expansion (Nel -> Stream)
      .filter { e => drop == 0 || e % 2 == 0 }               // dedup-style drop
      .map { e => Box(e) }                                   // yield
      .stateful(0) { (s, e) =>                               // monotonic cursor
        (Some(if (e.value > s) e.value else s), Stream[SyncIO].single(e))
      }
      .flatMapLast {                                         // Cassandra/Kafka seam
        case Some(b) => Stream[SyncIO].single(b)
        case None    => Stream[SyncIO].empty[Box]
      }

    stream.length.unsafeRunSync()
  }

  // Same pipeline under the production monad, to confirm the win survives cats.effect.IO.
  @Benchmark
  def pipelineIO(): Long = {
    val fo = fanout
    val drop = dropEvery
    val source: Stream[IO, Int] = Stream[IO].apply(records)

    val stream = source
      .flatMap { r =>
        if (r % 16 == 0) Stream[IO].empty[Int] else Stream[IO].single(r)
      }
      .filter { _ >= 0 }
      .flatMap { r => Stream.lift(IO(r)) }
      .flatMap { r => Stream[IO].apply(expand(r, fo)) }
      .filter { e => drop == 0 || e % 2 == 0 }
      .map { e => Box(e) }
      .stateful(0) { (s, e) =>
        (Some(if (e.value > s) e.value else s), Stream[IO].single(e))
      }
      .flatMapLast {
        case Some(b) => Stream[IO].single(b)
        case None    => Stream[IO].empty[Box]
      }

    stream.length.unsafeRunSync()
  }
}

object StreamCombinatorBenchmark {
  final case class Box(value: Int)
  private def expand(r: Int, fanout: Int): List[Int] = {
    val b = List.newBuilder[Int]
    var i = 0
    while (i < fanout) { b += r * 32 + i; i += 1 }
    b.result()
  }
}
