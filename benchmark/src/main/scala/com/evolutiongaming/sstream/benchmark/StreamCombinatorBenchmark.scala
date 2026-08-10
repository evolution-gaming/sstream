package com.evolutiongaming.sstream.benchmark

import cats.effect.unsafe.implicits.global
import cats.effect.{IO, SyncIO}
import cats.syntax.all.*
import com.evolutiongaming.sstream.Stream
import com.evolutiongaming.sstream.Stream.StreamOps
import org.openjdk.jmh.annotations.*

import java.util.concurrent.TimeUnit

/*
 * To run benchmarks:
 * {{{sbt benchmark/Jmh/run com.evolutiongaming.sstream.benchmark.StreamCombinatorBenchmark}}}
 * ------
 * Results on M3 Max (2023) (Oracle Corporation Java 17.0.11):
 * [info] Benchmark                                   (dropEvery)  (fanout)  (size)   Mode  Cnt    Score    Error  Units
 * [info] StreamCombinatorBenchmark.pipeline                    0         1   10000  thrpt    5  645.304 ±  3.694  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              0         1   10000  thrpt    5  558.189 ± 12.245  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    0         4   10000  thrpt    5  265.622 ±  9.840  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              0         4   10000  thrpt    5  218.824 ±  2.354  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    1         1   10000  thrpt    5  645.232 ±  6.819  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              1         1   10000  thrpt    5  549.975 ±  7.331  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    1         4   10000  thrpt    5  310.625 ±  4.306  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              1         4   10000  thrpt    5  269.069 ±  4.448  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  0         1   10000  thrpt    5  598.382 ± 47.818  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            0         1   10000  thrpt    5  510.236 ±  4.731  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  0         4   10000  thrpt    5  260.971 ±  1.739  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            0         4   10000  thrpt    5  210.391 ±  2.562  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  1         1   10000  thrpt    5  599.898 ±  8.612  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            1         1   10000  thrpt    5  493.247 ± 29.740  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  1         4   10000  thrpt    5  315.313 ±  3.703  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            1         4   10000  thrpt    5  267.250 ± 16.232  ops/s
 *
 * Results on M3 Max (2023) (Oracle Corporation Java 25):
 * [info] Benchmark                                   (dropEvery)  (fanout)  (size)   Mode  Cnt    Score    Error  Units
 * [info] StreamCombinatorBenchmark.pipeline                    0         1   10000  thrpt    5  744.755 ±  4.612  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              0         1   10000  thrpt    5  628.388 ± 10.069  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    0         4   10000  thrpt    5  297.407 ±  1.238  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              0         4   10000  thrpt    5  240.693 ±  1.908  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    1         1   10000  thrpt    5  725.814 ±  6.703  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              1         1   10000  thrpt    5  629.728 ±  3.180  ops/s
 * [info] StreamCombinatorBenchmark.pipeline                    1         4   10000  thrpt    5  361.284 ±  3.924  ops/s
 * [info] StreamCombinatorBenchmark.pipeline_1_1_0              1         4   10000  thrpt    5  310.471 ±  1.992  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  0         1   10000  thrpt    5  603.869 ±  2.266  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            0         1   10000  thrpt    5  524.131 ±  3.013  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  0         4   10000  thrpt    5  268.847 ±  3.404  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            0         4   10000  thrpt    5  228.081 ±  3.001  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  1         1   10000  thrpt    5  604.101 ±  1.952  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            1         1   10000  thrpt    5  530.051 ±  3.610  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO                  1         4   10000  thrpt    5  347.194 ±  2.417  ops/s
 * [info] StreamCombinatorBenchmark.pipelineIO_1_1_0            1         4   10000  thrpt    5  298.061 ±  2.877  ops/s
 *
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
      .flatMap { r => // record dispatch (single/empty)
        if (r % 16 == 0) Stream[SyncIO].empty[Int] else Stream[SyncIO].single(r)
      }
      .filter { _ >= 0 } // range guard
      .flatMap { r => Stream.lift(SyncIO(r)) } // effectful decode (F -> Stream)
      .flatMap { r => Stream[SyncIO].apply(expand(r, fo)) } // event expansion (Nel -> Stream)
      .filter { e => drop == 0 || e % 2 == 0 } // dedup-style drop
      .map { e => Box(e) } // yield
      .stateful(0) { (s, e) => // monotonic cursor
        (Some(if (e.value > s) e.value else s), Stream[SyncIO].single(e))
      }
      .flatMapLast { // Cassandra/Kafka seam
        case Some(b) => Stream[SyncIO].single(b)
        case None => Stream[SyncIO].empty[Box]
      }

    stream.length.unsafeRunSync()
  }

  @Benchmark
  def pipeline_1_1_0(): Long = {
    val fo = fanout
    val drop = dropEvery
    val source: Stream_1_1_0[SyncIO, Int] = Stream_1_1_0[SyncIO].apply(records)

    val stream = source
      .flatMap { r => // record dispatch (single/empty)
        if (r % 16 == 0) Stream_1_1_0[SyncIO].empty[Int] else Stream_1_1_0[SyncIO].single(r)
      }
      .filter { _ >= 0 } // range guard
      .flatMap { r => Stream_1_1_0.lift(SyncIO(r)) } // effectful decode (F -> Stream)
      .flatMap { r => Stream_1_1_0[SyncIO].apply(expand(r, fo)) } // event expansion (Nel -> Stream)
      .filter { e => drop == 0 || e % 2 == 0 } // dedup-style drop
      .map { e => Box(e) } // yield
      .stateful(0) { (s, e) => // monotonic cursor
        (Some(if (e.value > s) e.value else s), Stream_1_1_0[SyncIO].single(e))
      }
      .flatMapLast { // Cassandra/Kafka seam
        case Some(b) => Stream_1_1_0[SyncIO].single(b)
        case None => Stream_1_1_0[SyncIO].empty[Box]
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
        case None => Stream[IO].empty[Box]
      }

    stream.length.unsafeRunSync()
  }

  // Same pipeline under the production monad, to confirm the win survives cats.effect.IO.
  @Benchmark
  def pipelineIO_1_1_0(): Long = {
    val fo = fanout
    val drop = dropEvery
    val source: Stream_1_1_0[IO, Int] = Stream_1_1_0[IO].apply(records)

    val stream = source
      .flatMap { r =>
        if (r % 16 == 0) Stream_1_1_0[IO].empty[Int] else Stream_1_1_0[IO].single(r)
      }
      .filter { _ >= 0 }
      .flatMap { r => Stream_1_1_0.lift(IO(r)) }
      .flatMap { r => Stream_1_1_0[IO].apply(expand(r, fo)) }
      .filter { e => drop == 0 || e % 2 == 0 }
      .map { e => Box(e) }
      .stateful(0) { (s, e) =>
        (Some(if (e.value > s) e.value else s), Stream_1_1_0[IO].single(e))
      }
      .flatMapLast {
        case Some(b) => Stream_1_1_0[IO].single(b)
        case None => Stream_1_1_0[IO].empty[Box]
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
