import sbt._

object Dependencies {
  val scalatest              = "org.scalatest"              %% "scalatest"                 % "3.2.16"
  val discipline             = "org.typelevel"              %% "discipline-scalatest"      % "2.2.0"
  val `kind-projector`       = "org.typelevel"               % "kind-projector"            % "0.13.2"

  object Cats {
    private val version       = "2.7.0"
    private val effectVersion = "3.5.1"
    val core   = "org.typelevel" %% "cats-core"   % version
    val laws   = "org.typelevel" %% "cats-laws"   % version
    val effect = "org.typelevel" %% "cats-effect" % effectVersion
  }
}

