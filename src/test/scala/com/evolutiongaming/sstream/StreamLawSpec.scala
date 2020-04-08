package com.evolutiongaming.sstream


import cats.implicits._
import cats.kernel.laws.discipline.MonoidTests
import cats.laws.discipline.MonadTests
import cats.{Eq, Eval}
import org.scalacheck.Arbitrary
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.prop.Configuration
import org.typelevel.discipline.scalatest.FunSuiteDiscipline


class StreamLawSpec extends AnyFunSuite with FunSuiteDiscipline with Configuration {

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

  checkAll("Stream.MonadLaws", MonadTests[Stream[Eval, *]].stackUnsafeMonad[Int, Int, String])

  checkAll("Stream.MonoidLaws", MonoidTests[Stream[Eval, Int]].monoid)
}
