import sbt._

object Dependencies {
  val scalatest              = "org.scalatest"              %% "scalatest"                 % "3.2.0"
  val discipline             = "org.typelevel"              %% "discipline-scalatest"      % "1.0.1"
  val `scalacheck-shapeless` = "com.github.alexarchambault" %% "scalacheck-shapeless_1.14" % "1.2.5"
  val `kind-projector`       = "org.typelevel"               % "kind-projector"            % "0.10.3"

  object Cats {
    private val version = "2.1.1"
    val core   = "org.typelevel" %% "cats-core"   % version
    val laws   = "org.typelevel" %% "cats-laws"   % version
    val effect = "org.typelevel" %% "cats-effect" % "2.1.2"
  }
}