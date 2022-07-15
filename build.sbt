import Dependencies._

name := "sstream"

organization := "com.evolutiongaming"

homepage := Some(new URL("http://github.com/evolution-gaming/sstream"))

startYear := Some(2019)

organizationName := "Evolution"

organizationHomepage := Some(url("http://evolution.com"))

scalaVersion := crossScalaVersions.value.head

crossScalaVersions := Seq("2.13.4", "2.12.16")

publishTo := Some(Resolver.evolutionReleases)

libraryDependencies ++= Seq(
  Cats.core,
  Cats.effect,
  Cats.laws % Test,
  scalatest % Test,
  discipline % Test,
  `scalacheck-shapeless` % Test,
)

libraryDependencies += compilerPlugin(`kind-projector` cross CrossVersion.full)

licenses := Seq(("MIT", url("https://opensource.org/licenses/MIT")))

releaseCrossBuild := true

scalacOptsFailOnWarn := Some(false)