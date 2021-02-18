import sbt._

object Dependencies {
  val scalatest              = "org.scalatest"              %% "scalatest"                 % "3.2.3"
  val discipline             = "org.typelevel"              %% "discipline-scalatest"      % "2.1.2"
  val `scalacheck-shapeless` = "com.github.alexarchambault" %% "scalacheck-shapeless_1.14" % "1.2.5"
  val `kind-projector`       = "org.typelevel"               % "kind-projector"            % "0.11.2"

  object Cats {
    private val version = "2.3.0"
    val core   = "org.typelevel" %% "cats-core"   % version
    val laws   = "org.typelevel" %% "cats-laws"   % version
    val effect = "org.typelevel" %% "cats-effect" % version
  }
}