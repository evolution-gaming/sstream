package com.evolutiongaming.sstream

import cats.implicits._
import cats.{Foldable, Monad}

trait FoldWhile[F[_]] {

  def foldWhileM[G[_], A, L, R](fa: F[A], l: L)(f: (L, A) => G[Either[L, R]])(implicit G: Monad[G]): G[Either[L, R]]
}

object FoldWhile {

  implicit def foldWhileFoldable[F[_]](implicit F: Foldable[F]): FoldWhile[F] = new FoldWhile[F] {

    def foldWhileM[G[_], A, L, R](fa: F[A], l: L)(f: (L, A) => G[Either[L, R]])(implicit G: Monad[G]) = {
      F.foldLeftM[G, A, Either[L, R]](fa, l.asLeft[R]) {
        case (Left(l), a) => f(l, a)
        case (b, _)       => b.pure[G]
      }
    }
  }


  implicit class FoldWhileOps[F[_], A](val self: F[A]) extends AnyVal {

    def foldWhileM[G[_], L, R](l: L)(f: (L, A) => G[Either[L, R]])(implicit F: FoldWhile[F], G: Monad[G]): G[Either[L, R]] = {
      F.foldWhileM[G, A, L, R](self, l)(f)
    }
  }
}