# SStream
[![Build Status](https://github.com/evolution-gaming/sstream/workflows/CI/badge.svg)](https://github.com/evolution-gaming/sstream/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/evolution-gaming/sstream/badge.svg)](https://coveralls.io/r/evolution-gaming/sstream)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6ceee024f5c94cfa814e95675c77f2a9)](https://app.codacy.com/gh/evolution-gaming/sstream/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Version](https://img.shields.io/badge/version-click-blue)](https://evolution.jfrog.io/artifactory/api/search/latestVersion?g=com.evolutiongaming&a=sstream_2.13&repos=public)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://opensource.org/licenses/MIT)

## Use case examples

```scala
  test("take") {
    Stream[Id].many(1, 2, 3).take(1).toList shouldEqual List(1)
  }

  test("first") {
    Stream[Id].single(0).first shouldEqual Some(0)
  }

  test("repeat") {
    Stream.repeat[Id, Int](0).take(3).length shouldEqual 3
  }

  test("collect") {
    Stream[Id].many(1, 2, 3).collect { case x if x >= 2 => x + 1 }.toList shouldEqual List(3, 4)
  }

  test("zipWithIndex") {
    Stream.repeat[Id, Int](0).zipWithIndex.take(3).toList shouldEqual List((0, 0), (0, 1), (0, 2))
  }
```

## Setup

```scala
addSbtPlugin("com.evolution" % "sbt-artifactory-plugin" % "0.0.2")

libraryDependencies += "com.evolutiongaming" %% "sstream" % "0.2.1"
```

## Versioning

The library depends on Cats Effect thus published with version `1.X.X` for CE3 and `0.X.X` for CE2
