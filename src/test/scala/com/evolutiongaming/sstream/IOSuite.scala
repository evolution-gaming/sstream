package com.evolutiongaming.sstream

import cats.syntax.all._
import cats.effect.{IO, ContextShift, Timer}
import org.scalatest.Succeeded

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

object IOSuite {
  val Timeout: FiniteDuration = 5.seconds

  implicit val contextShiftIO: ContextShift[IO] = IO.contextShift(global)
  implicit val timerIO: Timer[IO]               = IO.timer(global)

  def runIO[A](io: IO[A], timeout: FiniteDuration = Timeout): Future[Succeeded.type] = {
    io.timeout(timeout).as(Succeeded).unsafeToFuture()
  }

  implicit class IOOps[A](val self: IO[A]) extends AnyVal {
    def run(timeout: FiniteDuration = Timeout): Future[Succeeded.type] = runIO(self, timeout)
  }
}