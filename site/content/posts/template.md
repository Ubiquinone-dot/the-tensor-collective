---
title: "Template"
date: .Date
author: ["Jasper Butcher"]
searchHidden: true

weight: 9999  # will be the last to be featured

cover:
    image: imgs/bug.png
    alt: cool
    caption: cool DALLE-3
TocOpen: false
ShowToc: true

tags: ["None", "Template", "ML"]
categories: ["ML"]
description: This is the description of the article, declared in the --- area of the MD file used to create it.
draft: false
---

# Showcase of the various features

## Code snippets
```python
def main(args):
    result = big_function(*args)
    return result
```
Or the below
```bash
ssh natrolite@chem.ox.ac.uk
```

## KaTeX

data! $e=mc$ neww $e=cm_{\sigma}$

Here's an inline equation for convenience.

Here is a more complex equation, written across a line:
$$
\int_{S\in R} U(S) dS \ \sim \mathcal{N}(0, \sigma^2I)
$$

This integral is central to many areas in mathematics.

## Images

None

## Basic ASCII diagrams
```goat
      .               .                .               .--- 1          .-- 1     / 1
     / \              |                |           .---+            .-+         +
    /   \         .---+---.         .--+--.        |   '--- 2      |   '-- 2   / \ 2
   +     +        |       |        |       |    ---+            ---+          +
  / \   / \     .-+-.   .-+-.     .+.     .+.      |   .--- 3      |   .-- 3   \ / 3
 /   \ /   \    |   |   |   |    |   |   |   |     '---+            '-+         +
 1   2 3   4    1   2   3   4    1   2   3   4         '--- 4          '-- 4     \ 4

```

See [here](https://gohugo.io/content-management/diagrams/#graphics) for more