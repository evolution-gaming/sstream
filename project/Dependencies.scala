import sbt.*

object Dependencies {
  val scalatest              = "org.scalatest"              %% "scalatest"                 % "3.2.19"
  val discipline             = "org.typelevel"              %% "discipline-scalatest"      % "2.3.0"
  val `kind-projector`       = "org.typelevel"               % "kind-projector"            % "0.13.3"

  object Cats {
    private val version       = "2.13.0"
    private val effectVersion = "3.7.0"
    val core   = "org.typelevel" %% "cats-core"   % version
    val laws   = "org.typelevel" %% "cats-laws"   % version
    val effect = "org.typelevel" %% "cats-effect" % effectVersion
  }
}

