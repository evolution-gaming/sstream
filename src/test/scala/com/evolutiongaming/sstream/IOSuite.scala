package com.evolutiongaming.sstream

import cats.effect.IO
import cats.effect.unsafe.IORuntime
import org.scalatest.Succeeded

import scala.concurrent.Future
import scala.concurrent.duration._

object IOSuite {
  val Timeout: FiniteDuration = 5.seconds

  implicit val ioRuntime: IORuntime = IORuntime.global

  def runIO[A](io: IO[A], timeout: FiniteDuration = Timeout): Future[Succeeded.type] = {
    io.timeout(timeout).as(Succeeded).unsafeToFuture()
  }

  implicit class IOOps[A](val self: IO[A]) extends AnyVal {
    def run(timeout: FiniteDuration = Timeout): Future[Succeeded.type] = runIO(self, timeout)
  }
}