import Dependencies._

name := "sstream"

organization := "com.evolutiongaming"

homepage := Some(new URL("http://github.com/evolution-gaming/sstream"))

startYear := Some(2019)

organizationName := "Evolution Gaming"

organizationHomepage := Some(url("http://evolutiongaming.com"))

bintrayOrganization := Some("evolutiongaming")

scalaVersion := crossScalaVersions.value.head

crossScalaVersions := Seq("2.13.0", "2.12.10")

resolvers += Resolver.bintrayRepo("evolutiongaming", "maven")

libraryDependencies ++= Seq(
  Cats.core,
  Cats.effect,
  Cats.laws % Test,
  scalatest % Test,
  discipline % Test,
  `scalacheck-shapeless` % Test,
)

libraryDependencies += compilerPlugin(`kind-projector` cross CrossVersion.binary)

licenses := Seq(("MIT", url("https://opensource.org/licenses/MIT")))

releaseCrossBuild := true