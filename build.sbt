import Dependencies._

name := "sstream"

organization := "com.evolutiongaming"

homepage := Some(new URL("http://github.com/evolution-gaming/sstream"))

startYear := Some(2019)

organizationName := "Evolution"

organizationHomepage := Some(url("http://evolution.com"))

scalaVersion := crossScalaVersions.value.head

crossScalaVersions := Seq("3.3.0", "2.13.11", "2.12.19")

publishTo := Some(Resolver.evolutionReleases)

libraryDependencies ++= Seq(
  Cats.core,
  Cats.effect,
  Cats.laws % Test,
  scalatest % Test,
  discipline % Test,
)

libraryDependencies ++= {
  if (scalaVersion.value.startsWith("3")) Nil
  else List(compilerPlugin(`kind-projector` cross CrossVersion.full))
}

ThisBuild / scalacOptions ++= {
  if (scalaVersion.value.startsWith("3")) Seq("-Ykind-projector:underscores")
  else Seq("-Xsource:3", "-P:kind-projector:underscore-placeholders")
}

licenses := Seq(("MIT", url("https://opensource.org/licenses/MIT")))

releaseCrossBuild := true

scalacOptsFailOnWarn := Some(false)
