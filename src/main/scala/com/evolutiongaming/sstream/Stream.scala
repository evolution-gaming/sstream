package com.evolutiongaming.sstream

import cats.effect.{Bracket, Resource, Sync}
import cats.implicits._
import cats.kernel.Monoid
import cats.{Applicative, FlatMap, Monad, StackSafeMonad, ~>}

import scala.util.{Left, Right}

trait Stream[F[_], A] {

  def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]): F[Either[L, R]]
}

object Stream { self =>

  def apply[F[_]](implicit F: Monad[F]): Builders[F] = new Builders[F](F)


  implicit def monadStream[F[_]]: Monad[Stream[F, *]] = new StackSafeMonad[Stream[F, *]] {

    def flatMap[A, B](fa: Stream[F, A])(f: A => Stream[F, B]) = fa.flatMap(f)

    def pure[A](a: A) = single[F, A](a)

    override def map[A, B](fa: Stream[F, A])(f: A => B) = fa.map(f)
  }


  implicit def monoidStream[F[_] : Monad, A]: Monoid[Stream[F, A]] = new Monoid[Stream[F, A]] {

    def empty = Stream.empty[F, A]

    def combine(x: Stream[F, A], y: Stream[F, A]) = x concat y
  }


  def lift[F[_], A](fa: F[A])(implicit monad: FlatMap[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = fa.flatMap(f(l, _))
  }

  def repeat[F[_], A](fa: F[A])(implicit F: Monad[F]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      for {
        r <- l.tailRecM { l =>
          for {
            a <- fa
            r <- f(l, a)
          } yield r
        }
      } yield {
        r.asRight
      }
    }
  }

  def single[F[_], A](a: A): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = f(l, a)
  }


  def from[F[_], G[_], A](ga: G[A])(implicit G: FoldWhile[G], monad: Monad[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = G.foldWhileM(ga, l)(f)
  }


  def empty[F[_], A](implicit F: Applicative[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = l.asLeft[R].pure[F]
  }


  def around[F[_]](f: F ~> F): Stream[F, Unit] = new Stream[F, Unit] {
    def foldWhileM[L, R](l: L)(f1: (L, Unit) => F[Either[L, R]]) = {
      f(f1(l, ()))
    }
  }


  def fromResource[F[_], A](resource: Resource[F, A])(implicit F: Bracket[F, Throwable]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      resource.use(a => f(l, a))
    }
  }


  def fromIterator[F[_] : Sync, A](iterator: F[Iterator[A]]): Stream[F, A] = {
    for {
      as <- Stream.lift(iterator)
      fa  = Sync[F].delay { if (as.hasNext) as.next().some else none[A] }
      a  <- untilNone(fa)
    } yield a
  }


  def untilNone[F[_] : Monad, A](a: F[Option[A]]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      l.tailRecM[F, Either[L, R]] { l =>
          for {
            a <- a
            a <- a.fold { l.asLeft[R].asRight[L].pure[F] } { a =>
              f(l, a).map {
                case a: Left[L, R]  => a.rightCast[Either[L, R]]
                case a: Right[L, R] => a.asRight[L]
              }
            }
          } yield a
        }
    }
  }


  final class Builders[F[_]](val F: Monad[F]) extends AnyVal {

    def apply[G[_], A](ga: G[A])(implicit G: FoldWhile[G]): Stream[F, A] = from[F, G, A](ga)(G, F)

    def apply[A](resource: Resource[F, A])(implicit F: Bracket[F, Throwable]): Stream[F, A] = {
      fromResource(resource)
    }

    def single[A](a: A): Stream[F, A] = Stream.single[F, A](a)

    def many[A](a: A, as: A*): Stream[F, A] = apply[List, A](a :: as.toList)

    def repeat[A](fa: F[A])(implicit F: Monad[F]): Stream[F, A] = self.repeat(fa)
  }


  sealed abstract class Cmd[+A] extends Product

  object Cmd {

    def take[A](value: A): Cmd[A] = Take(value)

    def stop[A]: Cmd[A] = Stop

    def skip[A]: Cmd[A] = Skip


    final case class Take[A] private(value: A) extends Cmd[A]

    final case object Skip extends Cmd[Nothing]

    final case object Stop extends Cmd[Nothing]
  }


  implicit class StreamOps[F[_], A](val self: Stream[F, A]) extends AnyVal {

    def mapK[G[_]](to: F ~> G, from: G ~> F): Stream[G, A] = new Stream[G, A] {

      def foldWhileM[L, R](l: L)(f: (L, A) => G[Either[L, R]]) = {
        to(self.foldWhileM(l) { (l, a) => from(f(l, a)) })
      }
    }

    def foldWhile[L, R](l: L)(f: (L, A) => Either[L, R])(implicit F: Applicative[F]): F[Either[L, R]] = {
      self.foldWhileM[L, R](l) { (l, a) => f(l, a).pure[F] }
    }


    def fold[B](b: B)(f: (B, A) => B)(implicit F: Applicative[F]): F[B] = {
      for {
        result <- foldWhile(b) { (b, a) => f(b, a).asLeft[B] }
      } yield {
        result.merge
      }
    }


    def foldM[B](b: B)(f: (B, A) => F[B])(implicit F: Applicative[F]): F[B] = {
      for {
        result <- self.foldWhileM(b) { (b, a) => f(b, a).map(_.asLeft[B]) }
      } yield {
        result.merge
      }
    }


    def toList(implicit F: Applicative[F]): F[List[A]] = {
      for {
        result <- fold(List.empty[A]) { (b, a) => a :: b }
      } yield {
        result.reverse
      }
    }


    def length(implicit F: Monad[F]): F[Long] = {
      fold(0L) { (n, _) => n + 1 }
    }


    def take(n: Long)(implicit F: Monad[F]): Stream[F, A] = {
      if (n <= 0) empty[F, A]
      else new Stream[F, A] {
        def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
          self
            .foldWhileM((l, n)) { case ((l, n), a) =>
              if (n == 0) {
                l.asLeft[R].asRight[(L, Long)].pure[F]
              } else if (n == 1) {
                f(l, a).map {
                  case a: Right[L, R] => a.asRight[(L, Long)]
                  case a: Left[L, R]  => a.asRight[(L, Long)]
                }
              } else {
                f(l, a).map {
                  case a: Right[L, R] => a.asRight[(L, Long)]
                  case Left(l)        => (l, n - 1).asLeft[Either[L, R]]
                }
              }
            }
            .map {
              case Left((a, _)) => a.asLeft[R]
              case Right(a)     => a
            }
        }
      }
    }


    def drop(n: Long)(implicit F: Monad[F]): Stream[F, A] = {
      if (n <= 0) self
      else new Stream[F, A] {
        def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
          self
            .foldWhileM((l, n)) { case ((l, n), a) =>
              if (n == 0) {
                f(l, a).map {
                  case a: Right[L, R] => a.asRight[(L, Long)]
                  case Left(l)        => (l, n).asLeft[Either[L, R]]
                }
              } else {
                (l, n - 1).asLeft[Either[L, R]].pure[F]
              }
            }
            .map {
              case Left((a, _)) => a.asLeft[R]
              case Right(a)     => a
            }
        }
      }
    }

    def first(implicit F: Applicative[F]): F[Option[A]] = {
      for {
        result <- foldWhile(none[A]) { (_, a) => a.some.asRight[Option[A]] }
      } yield {
        result.merge
      }
    }


    def last(implicit F: Applicative[F]): F[Option[A]] = {
      for {
        result <- foldWhile(none[A]) { (_, a) => a.some.asLeft[Option[A]] }
      } yield {
        result.merge
      }
    }


    def map[B](f: A => B): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self.foldWhileM(l) { (l, a) => f1(l, f(a)) }
      }
    }


    def mapM[B](f: A => F[B])(implicit F: FlatMap[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self.foldWhileM(l) { (l, a) => f(a).flatMap(b => f1(l, b)) }
      }
    }


    def flatMap[B](f: A => Stream[F, B]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self.foldWhileM(l) { (l, a) => f(a).foldWhileM(l)(f1) }
      }
    }


    def collect[B](pf: PartialFunction[A, B])(implicit F: Applicative[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f: (L, B) => F[Either[L, R]]) = {
        self.foldWhileM(l) { (l, a) => if (pf.isDefinedAt(a)) f(l, pf(a)) else l.asLeft[R].pure[F] }
      }
    }


    def filter(f: A => Boolean)(implicit F: Applicative[F]): Stream[F, A] = new Stream[F, A] {

      def foldWhileM[L, R](l: L)(f1: (L, A) => F[Either[L, R]]) = {
        self.foldWhileM(l) { (l, a) => if (f(a)) f1(l, a) else l.asLeft[R].pure[F] }
      }
    }


    def withFilter(f: A => Boolean)(implicit F: Applicative[F]): Stream[F, A] = filter(f)


    def concat[B >: A](stream: Stream[F, B])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f: (L, B) => F[Either[L, R]]) = {
        self.foldWhileM(l)(f).flatMap {
          case Left(l)        => stream.foldWhileM(l)(f)
          case a: Right[L, R] => a.leftCast[L].pure[F]
        }
      }
    }


    def zipWithIndex(implicit F: Monad[F]): Stream[F, (A, Long)] = {
      foldMap(0L) { (l, a) => (l + 1, (a, l)) }
    }


    def dropWhile(f: A => Boolean)(implicit F: Monad[F]): Stream[F, A] = {
      foldMapCmd(true) { (drop, a) => if (drop && f(a)) (drop, Cmd.skip) else (false, Cmd.take(a)) }
    }


    def takeWhile(f: A => Boolean)(implicit F: Monad[F]): Stream[F, A] = {
      mapCmd { a => if (f(a)) Cmd.take(a) else Cmd.stop }
    }


    def foldMapM[B, S](s: S)(f: (S, A) => F[(S, B)])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM((s, l)) { case ((s, l), a) =>
            for {
              ab     <- f(s, a)
              (s, b)  = ab
              result <- f1(l, b)
            } yield {
              result.leftMap { l => (s, l) }
            }
          }
          .map { _.leftMap { case (_, l) => l } }
      }
    }


    def foldMap[B, S](s: S)(f: (S, A) => (S, B))(implicit F: Monad[F]): Stream[F, B] = {
      foldMapM(s) { (s, a) => f(s, a).pure[F] }
    }


    def foldMapCmdM[B, S](s: S)(f: (S, A) => F[(S, Cmd[B])])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {

        self
          .foldWhileM((s, l)) { case ((s, l), a) =>
            for {
              ab       <- f(s, a)
              (s, cmd)  = ab
              result   <- cmd match {
                case Cmd.Skip         => (s, l).asLeft[Either[L, R]].pure[F]
                case Cmd.Stop         => l.asLeft[R].asRight[(S, L)].pure[F]
                case cmd: Cmd.Take[A] => for {
                  result <- f1(l, cmd.value)
                } yield result match {
                  case Left(l) => (s, l).asLeft[Either[L, R]]
                  case r       => r.asRight[(S, L)]
                }
              }
            } yield result
          }
          .map {
            case Left((_, l)) => l.asLeft[R]
            case Right(r)     => r
          }
      }
    }


    def foldMapCmd[B, S](s: S)(f: (S, A) => (S, Cmd[B]))(implicit F: Monad[F]): Stream[F, B] = {
      foldMapCmdM(s) { (s, a) => f(s, a).pure[F] }
    }


    def mapCmdM[B](f: A => F[Cmd[B]])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        for {
          result <- self.foldWhileM[L, Either[L, R]](l) { (l, a) =>
            for {
              cmd    <- f(a)
              result <- cmd match {
                case Cmd.Skip         => l.asLeft[Either[L, R]].pure[F]
                case Cmd.Stop         => l.asLeft[R].asRight[L].pure[F]
                case cmd: Cmd.Take[B] => for {
                  result <- f1(l, cmd.value)
                } yield result match {
                  case a: Left[L, R] => a.rightCast[Either[L, R]]
                  case a             => a.asRight[L]
                }
              }
            } yield result
          }
        } yield {
          result.joinRight
        }
      }
    }


    def mapCmd[B](f: A => Cmd[B])(implicit F: Monad[F]): Stream[F, B] = {
      mapCmdM { a => f(a).pure[F] }
    }


    def drain(implicit F: Applicative[F]): F[Unit] = {
      val unit = ().asLeft[Unit].pure[F]
      self
        .foldWhileM(()) { (_, _) => unit }
        .map(_.merge)
    }
  }


  implicit class FlattenOps[F[_], A](val self: Stream[F, Stream[F, A]]) extends AnyVal {

    def flatten: Stream[F, A] = self.flatMap(identity)
  }
}
