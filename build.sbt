import Dependencies.*

name := "sstream"

organization := "com.evolutiongaming"

homepage := Some(uri("https://github.com/evolution-gaming/sstream"))

startYear := Some(2019)

organizationName := "Evolution"

organizationHomepage := Some(uri("https://evolution.com"))

scalaVersion := crossScalaVersions.value.head

crossScalaVersions := Seq("3.3.8", "3.9.0")

publishTo := Some(Resolver.evolutionReleases)

versionPolicyIntention := Compatibility.BinaryCompatible

libraryDependencies ++= Seq(
  Cats.core,
  Cats.effect,
  Cats.laws % Test,
  scalatest % Test,
  discipline % Test,
)

ThisBuild / libraryDependencies ++= {
  if (scalaVersion.value.startsWith("3")) Nil
  else Seq(compilerPlugin(`kind-projector`.cross(CrossVersion.full)))
}

ThisBuild / scalacOptions ++= {
  scalaBinaryVersion.value match {
    case "2.13" =>
      Seq(
        "-Xsource:3-cross",
        "-P:kind-projector:underscore-placeholders",
      )
    case _ =>
      Seq(
        "-Ykind-projector:underscores",
        // improve error messages:
        "-explain",
        "-explain-types",
      )
  }
}

licenses := Seq(("MIT", uri("https://opensource.org/licenses/MIT")))

scalacOptsFailOnWarn := Some(false)

addCommandAlias("check", "+all scalafmtCheckRepo versionPolicyCheck Compile/doc")
addCommandAlias("fmt", "+scalafmtRepo")
addCommandAlias("build", "+all compile testFull")

lazy val benchmark = (project in file("benchmark"))
  .enablePlugins(JmhPlugin)
  .dependsOn(LocalRootProject)
  .settings(
    name := "sstream-benchmark",
    scalaVersion := (LocalRootProject / scalaVersion).value,
    crossScalaVersions := (LocalRootProject / crossScalaVersions).value,
    publish / skip := true,
    Compile / doc / sources := Seq.empty,
    scalacOptions := (LocalRootProject / scalacOptions).value,
    Compile / unmanagedSourceDirectories := {
      if (scalaVersion.value.startsWith("3")) (Compile / unmanagedSourceDirectories).value
      else Seq.empty
    },
  )
