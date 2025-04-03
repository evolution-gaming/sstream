package com.evolutiongaming.sstream

import cats.effect.{MonadCancel, Resource, Sync}
import cats.kernel.Monoid
import cats.syntax.all.*
import cats.{Applicative, ApplicativeError, FlatMap, Functor, Monad, StackSafeMonad, ~>}

import scala.util.{Left, Right}

/**
 * Super simple interface for describing streaming capabilities.
 * Used in API description.
 * You might end up with gluing this with akka-streams or FS2 in attempt to gain more power and many more combinators
 */
trait Stream[F[_], A] {

  /** Takes initial `L` and combines with all `A` until `Right[R]` is returned
    * by `f`.
    *
    * The stream calls `f` over and over with a new `A` value until either `f`
    * returns `Right[R]` or there are no more `A` left in a stream. Each
    * consecutive call to `f` will use `l` returned by a previous call to `f`.
    *
    * @return
    *   `Right[R]` if `f` returned `Right[R]` or `Left[L]` if it never did, and
    *   there are no more `A` values left.
    */
  def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]): F[Either[L, R]]
}

object Stream { self =>

  def apply[F[_]](implicit F: Monad[F]): Builders[F] = new Builders[F](F)


  implicit def monadStream[F[_]]: Monad[Stream[F, _]] = new StackSafeMonad[Stream[F, _]] {

    def flatMap[A, B](fa: Stream[F, A])(f: A => Stream[F, B]) = fa.flatMap(f)

    def pure[A](a: A) = single[F, A](a)

    override def map[A, B](fa: Stream[F, A])(f: A => B) = fa.map(f)
  }


  implicit def monoidStream[F[_] : Monad, A]: Monoid[Stream[F, A]] = new Monoid[Stream[F, A]] {

    def empty = Stream.empty

    def combine(x: Stream[F, A], y: Stream[F, A]) = x concat y
  }

  /** Stream of a single effectful `A` value.
    *
    * Example:
    * {{{
    * scala> import cats.effect.IO
    * scala> import cats.effect.unsafe.implicits.global
    * scala> import com.evolutiongaming.sstream.Stream
    * scala> import scala.util.Random
    *
    * scala> Stream
    *        .lift(IO(Random.nextInt(5)))
    *        .toList
    *        .unsafeRunSync()
    * val res0: List[Int] = List(3)
    * }}}
    */
  def lift[F[_], A](fa: F[A])(implicit monad: FlatMap[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = fa.flatMap(f(l, _))
  }

  /** Infinite stream of effectful `A` values.
    *
    * Example:
    * {{{
    * scala> import cats.effect.IO
    * scala> import cats.effect.unsafe.implicits.global
    * scala> import com.evolutiongaming.sstream.Stream
    * scala> import scala.util.Random
    *
    * scala> Stream
    *        .repeat(IO(Random.nextInt(5)))
    *        .take(10)
    *        .toList
    *        .unsafeRunSync()
    * val res0: List[Int] = List(1, 3, 0, 1, 0, 4, 1, 2, 0, 0)
    * }}}
    *
    * @see [[Builders#repeat]] for a more convenient syntax.
    */
  def repeat[F[_], A](fa: F[A])(implicit F: Monad[F]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      l
        .tailRecM { l =>
          for {
            a <- fa
            r <- f(l, a)
          } yield r
        }
        .map { _.asRight }
    }
  }

  /** Stream with a single element `A`.
    *
    * Example:
    * {{{
    * scala> import cats.Id
    * scala> import com.evolutiongaming.sstream.Stream
    *
    * scala> Stream.single[Id, Int](123).toList
    * val res0: List[Int] = List(123)
    * }}}
    *
    * @see [[Builders#single]] for a more convenient syntax.
    */
  def single[F[_], A](a: A): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = f(l, a)
  }

  /** Create a stream from any foldable structure.
    *
    * Example:
    * {{{
    * scala> import cats.Id
    * scala> import com.evolutiongaming.sstream.Stream
    *
    * scala> Stream.from[Id, Vector, Int](Vector(1, 2, 3, 4)).toList
    * val res0: List[Int] = List(1, 2, 3, 4)
    * }}}
    */
  def from[F[_], G[_], A](ga: G[A])(implicit G: FoldWhile[G], monad: Monad[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = G.foldWhileM(ga, l)(f)
  }


  /** Empty stream containing no elements */
  def empty[F[_], A](implicit F: Applicative[F]): Stream[F, A] = new Stream[F, A] {
    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = l.asLeft[R].pure[F]
  }

  /** Same as `Stream.single(())`, but applies a transformation on an effect.
    *
    * It could be useful to add an error handling or retry mechanics to an
    * existing stream.
    *
    * Example (print every element 3 times):
    * {{{
    * scala> import cats.arrow.FunctionK
    * scala> import cats.effect.IO
    * scala> import cats.effect.unsafe.implicits.global
    * scala> import com.evolutiongaming.sstream.Stream
    * scala> import scala.util.Random
    *
    * scala> val repeat3 = new FunctionK[IO, IO] {
    *          def apply[A](fa: IO[A]): IO[A] = fa *> fa *> fa
    *        }
    * scala> val stream = for {
    *          n <- Stream[IO].many(3, 2, 1, 0)
    *          _ <- Stream.around(repeat3)
    *          _ <- Stream.lift(IO.print(s"\$n "))
    *        } yield n
    * 3 3 3 2 2 2 1 1 1 0 0 0
    * val res0: List[Int] = List(3, 2, 1, 0)
    * }}}
    *
    * @see
    *   [[https://github.com/evolution-gaming/retry]] library for an out of the
    *   box retry builders to use as a parameter for this function.
    */
  def around[F[_]](f: F ~> F): Stream[F, Unit] = new Stream[F, Unit] {
    def foldWhileM[L, R](l: L)(f1: (L, Unit) => F[Either[L, R]]) = f(f1(l, ()))
  }


  def fromResource[F[_], A](resource: Resource[F, A])(implicit F: MonadCancel[F, Throwable]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      resource.use(a => f(l, a))
    }
  }


  def fromIterator[F[_] : Sync, A](iterator: F[Iterator[A]]): Stream[F, A] = {
    for {
      as <- Stream.lift(iterator)
      a   = Sync[F].delay { if (as.hasNext) as.next().some else none[A] }
      a  <- whileSome(a)
    } yield a
  }


  def whileSome[F[_] : Monad, A](a: F[Option[A]]): Stream[F, A] = new Stream[F, A] {

    def foldWhileM[L, R](l: L)(f: (L, A) => F[Either[L, R]]) = {
      l.tailRecM[F, Either[L, R]] { l =>
        for {
          a <- a
          a <- a.fold {
            l.asLeft[R].asRight[L].pure[F]
          } { a =>
            f(l, a).map {
              case l: Left[L, R]  => l.rightCast[Either[L, R]]
              case r: Right[L, R] => r.asRight[L]
            }
          }
        } yield a
      }
    }
  }


  @deprecated("use whileSome instead", "0.1.0")
  def untilNone[F[_] : Monad, A](a: F[Option[A]]): Stream[F, A] = whileSome(a)


  final class Builders[F[_]](val F: Monad[F]) extends AnyVal {

    def apply[G[_], A](ga: G[A])(implicit G: FoldWhile[G]): Stream[F, A] = from[F, G, A](ga)(G, F)

    def apply[A](resource: Resource[F, A])(implicit F: MonadCancel[F, Throwable]): Stream[F, A] = {
      fromResource(resource)
    }

    def empty[A]: Stream[F, A] = Stream.empty(F)

    def single[A](a: A): Stream[F, A] = Stream.single[F, A](a)

    def many[A](a: A, as: A*): Stream[F, A] = apply[List, A](a :: as.toList)

    def repeat[A](fa: F[A])(implicit F: Monad[F]): Stream[F, A] = self.repeat(fa)
  }


  @deprecated("Use `statefulM` instead", "0.0.10")
  sealed abstract class Cmd[+A] extends Product

  object Cmd {

    def take[A](value: A): Cmd[A] = Take(value)

    def stop[A]: Cmd[A] = Stop

    def skip[A]: Cmd[A] = Skip


    final case class Take[+A] private[Cmd](value: A) extends Cmd[A]

    case object Skip extends Cmd[Nothing]

    case object Stop extends Cmd[Nothing]
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
      if (n <= 0) empty
      else stateful(n) { (n, a) =>
        if (n == 0) {
          (none[Long], empty)
        } else if (n == 1) {
          (none[Long], single(a))
        } else {
          ((n - 1).some, single(a))
        }
      }
    }


    def drop(n: Long)(implicit F: Monad[F]): Stream[F, A] = {
      if (n <= 0) self
      else stateful(n) { (n, a) =>
        if (n == 0) {
          (n.some, single(a))
        } else {
          ((n - 1).some, empty)
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
        self
          .foldWhileM(l)(f)
          .flatMap {
            case Left(l)        => stream.foldWhileM(l)(f)
            case r: Right[L, R] => r.leftCast[L].pure[F]
          }
      }
    }


    def handleErrorWith[E](f: E => Stream[F, A])(implicit F: ApplicativeError[F, E]): Stream[F, A] = new Stream[F, A] {

      def foldWhileM[L, R](l: L)(f1: (L, A) => F[Either[L, R]]) = {
        self
          .foldWhileM(l)(f1)
          .handleErrorWith { a => f(a).foldWhileM(l)(f1) }
      }
    }


    def zipWithIndex(implicit F: Monad[F]): Stream[F, (A, Long)] = {
      foldMap(0L) { (l, a) => (l + 1, (a, l)) }
    }


    def dropWhile(f: A => Boolean)(implicit F: Monad[F]): Stream[F, A] = {
      stateful(true) { (drop, a) =>
        if (drop && f(a)) (drop.some, empty)
        else (false.some, single(a))
      }
    }

    def takeWhile(f: A => Boolean)(implicit F: Monad[F]): Stream[F, A] = {
      stateless { a =>
        if (f(a)) (true, single(a))
        else (false, empty)
      }
    }


    def foldMapM[B, S](s: S)(f: (S, A) => F[(S, B)])(implicit F: Monad[F]): Stream[F, B] = {
      statefulM(s) { (s, a) =>
        f(s, a).map { case (s, b) =>
          (s.some, single(b))
        }
      }
    }

    def foldMap[B, S](s: S)(f: (S, A) => (S, B))(implicit F: Functor[F]): Stream[F, B] = {
      stateful(s) { (s, a) =>
        val (s1, b) = f(s, a)
        (s1.some, single(b))
      }
    }


    @deprecated("Use `statefulM` instead", "0.0.10")
    def foldMapCmdM[B, S](s: S)(f: (S, A) => F[(S, Cmd[B])])(implicit F: Monad[F]): Stream[F, B] = {
      statefulM(s) { (s, a) =>
        f(s, a).map { case (s, c) =>
          c match {
            case c: Cmd.Take[B] => (s.some, single(c.value))
            case Cmd.Skip       => (s.some, empty)
            case Cmd.Stop       => (none[S], empty)
          }
        }
      }
    }


    @deprecated("Use `statefulM` instead", "0.0.10")
    def foldMapCmd[B, S](s: S)(f: (S, A) => (S, Cmd[B]))(implicit F: Monad[F]): Stream[F, B] = {
      foldMapCmdM(s) { (s, a) => f(s, a).pure[F] }
    }


    @deprecated("Use `statefulM` instead", "0.0.10")
    def mapCmdM[B](f: A => F[Cmd[B]])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        for {
          result <- self.foldWhileM[L, Either[L, R]](l) { (l, a) =>
            for {
              cmd    <- f(a)
              result <- cmd match {
                case Cmd.Skip         => l.asLeft[Either[L, R]].pure[F]
                case Cmd.Stop         => l.asLeft[R].asRight[L].pure[F]
                case cmd: Cmd.Take[B] => f1(l, cmd.value).map {
                  case l: Left[L, R]  => l.rightCast[Either[L, R]]
                  case r: Right[L, R] => r.asRight[L]
                }
              }
            } yield result
          }
        } yield {
          result.joinRight
        }
      }
    }


    @deprecated("Use `statefulM` instead", "0.0.10")
    def mapCmd[B](f: A => Cmd[B])(implicit F: Monad[F]): Stream[F, B] = {
      mapCmdM { a => f(a).pure[F] }
    }


    def drain(implicit F: Applicative[F]): F[Unit] = {
      val unit = ().asLeft[Unit].pure[F]
      self
        .foldWhileM(()) { (_, _) => unit }
        .map(_.merge)
    }


    def foreach(f: A => F[Unit])(implicit F: Functor[F]): F[Unit] = {
      val unit = ().asLeft[Unit]
      self
        .foldWhileM(()) { case (_, a) => f(a) as unit }
        .map(_.merge)
    }


    /** Similar to [[flatMap]], but allows keeping a state or shortcut the
      * computation.
      *
      * Example (make upper case out of symbols until 3 symbols are gathered):
      * {{{
      * scala> import cats.Id
      * scala> import com.evolutiongaming.sstream.Stream
      *
      * scala> Stream[Id]
      *        .many("a", "b", "c", "d", "e")
      *        .stateful(1) { (count, a) =>
      *          val state = Option.when(count < 3)(count + 1)
      *          val output = Stream[Id].single(a.toUpperCase)
      *          (state, output)
      *        }
      *        .toList
      * val res0: List[String] = List(A, B, C)
      * }}}
      *
      * @param s
      *   The initial state.
      * @param f
      *   Converts previous state and a new input element, to a new state and a
      *   stream of output elements. The stream is finished if there are no more
      *   elements in original stream, or if this function returns `None` as a
      *   state.
      *
      * @return
      *   Flattened stream of elements returned by `f`.
      *
      * @see
      *   [[stateless]] if only shortcut semantics is required.
      */
    def stateful[S, B](
      s: S)(
      f: (S, A) => (Option[S], Stream[F, B]))(implicit
      F: Functor[F]
    ): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM((l, s)) { case ((l, s0), a) =>
            val (s, stream) = f(s0, a)
            stream
              .foldWhileM(l)(f1)
              .map {
                case Left(l)        => s match {
                  case Some(s) => (l, s).asLeft[Either[L, R]]
                  case None    => l.asLeft[R].asRight[(L, S)]
                }
                case r: Right[L, R] => r.asRight[(L, S)]
              }
          }
          .map {
            case Left((l, _)) => l.asLeft[R]
            case Right(r)     => r
          }
      }
    }

    /** Same as [[stateful]], but allows `f` calculation to be effectful */
    def statefulM[S, B](
      s: S)(
      f: (S, A) => F[(Option[S], Stream[F, B])])(implicit
      F: FlatMap[F]
    ): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM((l, s)) { case ((l, s), a) =>
            f(s, a).flatMap { case (s, stream) =>
              stream
                .foldWhileM(l)(f1)
                .map {
                  case Left(l)        => s match {
                    case Some(s) => (l, s).asLeft[Either[L, R]]
                    case None    => l.asLeft[R].asRight[(L, S)]
                  }
                  case r: Right[L, R] => r.asRight[(L, S)]
                }
            }
          }
          .map {
            case Left((l, _)) => l.asLeft[R]
            case Right(r)     => r
          }
      }
    }

    /** Similar to [[flatMap]], but allows to shortcut the computation.
      *
      * Example (make upper case out of symbols until "c" is encountered):
      * {{{
      * scala> import cats.Id
      * scala> import com.evolutiongaming.sstream.Stream
      *
      * scala> Stream[Id]
      *        .many("a", "b", "c", "d", "e")
      *        .stateless { a =>
      *          val continue = (a != "c")
      *          val output = Stream[Id].single(a.toUpperCase)
      *          (continue, output)
      *        }
      *        .toList
      * val res0: List[String] = List(A, B, C)
      * }}}
      *
      * @param f
      *   Converts a new input element, to `continue` flag and a stream of
      *   output elements. The stream is finished if there are no more elements
      *   in original stream, or if this function returns (`false`, _) as a
      *   resulting tuple.
      *
      * @return
      *   Flattened stream of elements returned by `f`.
      *
      * @see
      *   [[stateful]] if stateful processing, in addition to shortcut
      *   semantics, is required.
      */
    def stateless[B](
      f: A => (Boolean, Stream[F, B]))(implicit
      F: Functor[F]
    ): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM(l) { (l, a) =>
            val (continue, stream) = f(a)
            stream
              .foldWhileM(l)(f1)
              .map {
                case l: Left[L, R]  => if (continue) l.rightCast[Either[L, R]] else l.asRight[L]
                case r: Right[L, R] => r.asRight[L]
              }
          }
          .map {
            case l: Left[L, Either[L, R]] => l.rightCast[R]
            case Right(r)                 => r
          }
      }
    }

    /** Same as [[stateless]], but allows `f` calculation to be effectful */
    def statelessM[B](
      f: A => F[(Boolean, Stream[F, B])])(implicit
      F: FlatMap[F]
    ): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM(l) { (l, a) =>
            f(a).flatMap { case (continue, stream) =>
              stream
                .foldWhileM(l)(f1)
                .map {
                  case l: Left[L, R]  => if (continue) l.rightCast[Either[L, R]] else l.asRight[L]
                  case r: Right[L, R] => r.asRight[L]
                }
            }
          }
          .map {
            case l: Left[L, Either[L, R]] => l.rightCast[R]
            case Right(r)                 => r
          }
      }
    }


    def foldLeftM[B](b: B)(f: (B, A) => F[B])(implicit F: FlatMap[F]): Stream[F, B] = {
      statefulM(b) { (b, a) =>
        f(b, a).map { b =>
          (b.some, Stream.single[F, B](b))
        }
      }
    }


    def foldLeft[B](b: B)(f: (B, A) => B)(implicit F: Functor[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM((l, b)) { case ((l, b), a) =>
            val b1 = f(b, a)
            f1(l, b1).map {
              case Left(l)        => (l, b1).asLeft[R]
              case r: Right[L, R] => r.leftCast[(L, B)]
            }
          }
          .map { _.leftMap { case (l, _) => l } }
      }
    }

    /** After this stream is exhausted, stream elements from another stream.
      *
      * Similar to [[concat]], but allows to use the last element emitted by a
      * first stream to construct the second one.
      *
      * Example (multiply last element by 2, 3 and 4):
      * {{{
      * scala> import cats.Id
      * scala> import com.evolutiongaming.sstream.Stream
      *
      * scala> Stream[Id]
      *        .many(1, 2, 3, 4, 5)
      *        .flatMapLast { a =>
      *          val factor = a.getOrElse(1)
      *          Stream[Id].many(factor * 2, factor * 3, factor * 4)
      *        }
      *        .toList
      * val res0: List[String] = List(1, 2, 3, 4, 5, 10, 15, 20)
      * }}}
      *
      * @param f
      *   Converts a last stream element, to a stream of output elements to be
      *   emited after all elements of original stream are emitted. If the
      *   original stream had no elements then `None` is passed to `f` instead.
      *
      * @return
      *   Stream the same elements as original stream does and then stream the
      *   elements returned by `f`.
      */
    def flatMapLast[B >: A](f: Option[A] => Stream[F, B])(implicit F: Monad[F]): Stream[F, B] = new Stream[F, B] {

      def foldWhileM[L, R](l: L)(f1: (L, B) => F[Either[L, R]]) = {
        self
          .foldWhileM((l, none[A])) { case ((l, _), a) =>
            f1(l, a).map {
              case Left(l)        => (l, a.some).asLeft[R]
              case a: Right[L, R] => a.leftCast[(L, Option[A])]
            }
          }
          .flatMap {
            case Left((l, a))                => f(a).foldWhileM(l)(f1)
            case a: Right[(L, Option[A]), R] => a.leftCast[L].pure[F]
          }
      }
    }

    /** Apply a function recursively on the last element of the stream.
      *
      * Similar to [[flatMapLast]], but also applies itself to the produced elements
      * until `None` is returned by `f`.
      *
      * Example (increment last element until it becomes 35):
      * {{{
      * scala> import cats.Id
      * scala> import com.evolutiongaming.sstream.Stream
      *
      * scala> Stream[Id]
      *        .many(10, 20, 30)
      *        .chain { a =>
      *          Option.when(a < 35) {
      *            Stream[Id].single(a + 1)
      *          }
      *        }
      *        .toList
      * val res0: List[String] = List(10, 20, 30, 31, 32, 33, 34, 35)
      * }}}
      *
      * @param f
      *   Converts a last stream element, to a stream of output elements to be
      *   emited after all elements of original stream are emitted. Also applies
      *   the same function to the new last element until the function returns
      *   `None`.
      *
      * @return
      *   Stream the same elements as original stream does, then stream the
      *   elements returned by recursive application of `f` to last element
      *   and last elements of the streams produced by itself.
      */
    def chain(f: A => Option[Stream[F, A]])(implicit F: Monad[F]): Stream[F, A] = {
      chainM { a => f(a).pure[F] }
    }

    /** Same as [[chain]], but allows `f` calculation to be effectful */
    def chainM(f: A => F[Option[Stream[F, A]]])(implicit F: Monad[F]): Stream[F, A] = new Stream[F, A] {

      def foldWhileM[L, R](l: L)(f1: (L, A) => F[Either[L, R]]) = {

        def drain(stream: Stream[F, A], l: L) = {
          stream.foldWhileM((l, none[A])) { case ((l, _), a) =>
            f1(l, a).map { _.leftMap { l => (l, a.some) } }
          }
        }

        drain(self, l).flatMap {
          case Left((l, Some(a)))          =>
            (l, a).tailRecM[F, Either[L, R]] { case (l, a) =>
              f(a).flatMap {
                case Some(stream) => drain(stream, l).map {
                  case Left((l, Some(a)))          => (l, a).asLeft[Either[L, R]]
                  case Left((l, None))             => l.asLeft[R].asRight[(L, A)]
                  case r: Right[(L, Option[A]), R] => r.leftCast[L].asRight[(L, A)]
                }
                case None         => l.asLeft[R].asRight[(L, A)].pure[F]
              }
            }
          case Left((l, None))             => l.asLeft[R].pure[F]
          case r: Right[(L, Option[A]), R] => r.leftCast[L].pure[F]
        }
      }
    }
  }


  implicit class FlattenOps[F[_], A](val self: Stream[F, Stream[F, A]]) extends AnyVal {

    def flatten: Stream[F, A] = self.flatMap(identity)
  }
}
