import sbt._

object Dependencies {
  val scalatest              = "org.scalatest"              %% "scalatest"                 % "3.0.8"
  val `scalacheck-shapeless` = "com.github.alexarchambault" %% "scalacheck-shapeless_1.13" % "1.1.8"
  val `kind-projector`       = "org.typelevel"               % "kind-projector"            % "0.10.3"

  object Cats {
    private val version = "1.6.1"
    val core   = "org.typelevel" %% "cats-core"   % version
    val laws   = "org.typelevel" %% "cats-laws"   % version
    val effect = "org.typelevel" %% "cats-effect" % "1.4.0"
  }
}