# SStream
[![Build Status](https://github.com/evolution-gaming/sstream/workflows/CI/badge.svg)](https://github.com/evolution-gaming/sstream/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/evolution-gaming/sstream/badge.svg)](https://coveralls.io/r/evolution-gaming/sstream)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/19db076d4ff64a78b865a17354144e9a)](https://www.codacy.com/app/evolution-gaming/sstream?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=evolution-gaming/sstream&amp;utm_campaign=Badge_Grade)
[![version](https://api.bintray.com/packages/evolutiongaming/maven/sstream/images/download.svg) ](https://bintray.com/evolutiongaming/maven/sstream/_latestVersion)
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
resolvers += Resolver.bintrayRepo("evolutiongaming", "maven")

libraryDependencies += "com.evolutiongaming" %% "sstream" % "0.0.1"
```