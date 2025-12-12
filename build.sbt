import Dependencies.*

name := "sstream"

organization := "com.evolutiongaming"

homepage := Some(url("https://github.com/evolution-gaming/sstream"))

startYear := Some(2019)

organizationName := "Evolution"

organizationHomepage := Some(url("https://evolution.com"))

scalaVersion := crossScalaVersions.value.head

crossScalaVersions := Seq("3.3.5", "2.13.16", "2.12.21")

publishTo := Some(Resolver.evolutionReleases)

versionPolicyIntention := Compatibility.BinaryCompatible

libraryDependencies ++= Seq(
  Cats.core,
  Cats.effect,
  Cats.laws % Test,
  scalatest % Test,
  discipline % Test,
)

libraryDependencies ++= {
  if (scalaVersion.value.startsWith("3")) Nil
  else Seq(compilerPlugin(`kind-projector` cross CrossVersion.full))
}

ThisBuild / scalacOptions ++= {
  if (scalaVersion.value.startsWith("3")) Seq(
    "-Ykind-projector:underscores",
  ) else if (scalaVersion.value.startsWith("2.12")) Seq(
    "-Xsource:3",
    "-P:kind-projector:underscore-placeholders",
  ) else Seq( // 2.13.x
    "-Xsource:3-cross",
    "-P:kind-projector:underscore-placeholders",
  )
}

licenses := Seq(("MIT", url("https://opensource.org/licenses/MIT")))

scalacOptsFailOnWarn := Some(false)

addCommandAlias("check", "+all versionPolicyCheck Compile/doc")
addCommandAlias("build", "+all compile test")
