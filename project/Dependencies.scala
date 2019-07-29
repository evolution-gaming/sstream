import sbt._

object Dependencies {
  val scalatest = "org.scalatest" %% "scalatest" % "3.0.8"

  object Cats {
    private val version = "1.6.1"
    val core   = "org.typelevel" %% "cats-core"   % version
    val effect = "org.typelevel" %% "cats-effect" % "1.3.1"
  }
}