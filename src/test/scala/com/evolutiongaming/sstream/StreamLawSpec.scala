package com.evolutiongaming.sstream


import cats.implicits._
import cats.laws.discipline.MonadTests
import cats.{Eq, Eval}
import org.scalacheck.Arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.typelevel.discipline.scalatest.Discipline


class StreamLawSpec extends AnyFunSuite with Discipline {

  implicit def eqStream[A: Eq]: Eq[Stream[Eval, A]] = (x: Stream[Eval, A], y: Stream[Eval, A]) => {
    x.toList.value == y.toList.value
  }

  implicit def arbStream[A: Arbitrary]: Arbitrary[Stream[Eval, A]] =
    Arbitrary {
      for {
        a <- Arbitrary.arbitrary[A]
      } yield {
        Stream[Eval].single(a)
      }
    }

  checkAll("Stream.MonadLaws", MonadTests[Stream[Eval, ?]].stackUnsafeMonad[Int, Int, String])
}
