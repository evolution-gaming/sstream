package com.evolutiongaming.sstream

import cats.effect.{Concurrent, ContextShift, IO, Timer}
import cats.implicits._
import org.scalatest.Succeeded

import scala.concurrent.duration._
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}

object IOSuite {
  val Timeout: FiniteDuration = 5.seconds

  implicit val executor: ExecutionContextExecutor = ExecutionContext.global

  implicit val contextShiftIO: ContextShift[IO]     = IO.contextShift(executor)
  implicit val concurrentIO: Concurrent[IO]         = IO.ioConcurrentEffect
  implicit val timerIO: Timer[IO]                   = IO.timer(executor)

  def runIO[A](io: IO[A], timeout: FiniteDuration = Timeout): Future[Succeeded.type] = {
    io.timeout(timeout).as(Succeeded).unsafeToFuture
  }

  implicit class IOOps[A](val self: IO[A]) extends AnyVal {
    def run(timeout: FiniteDuration = Timeout): Future[Succeeded.type] = runIO(self, timeout)
  }
}