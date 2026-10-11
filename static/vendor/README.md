# Vendored browser libraries

Copied byte for byte from the npm package tarballs, whose sha512 matched the
registry's `dist.integrity`. The page loads them with the `integrity` (SRI)
values below, so a modified file is refused by the browser.

| File | Package | npm integrity of the tarball | SRI of the file |
| --- | --- | --- | --- |
| `marked-18.0.14.umd.js` | marked 18.0.14, `lib/marked.umd.js` (MIT, `LICENSE.marked`) | `sha512-mBHK6FBHuBAlhgRe88w9F0O1AbwwXJUcQibUbC/QcdTbVGAD7aWza+xt3N6oT/jCZx3/OMeS+8rnuiHZcQ9s7A==` | `sha384-2vpGtuKqJvFlwJqYnf/wUMuzUfhUnYBt9oay0e2yaFcq0Dh6/aEbQ8YAOeKGzlYo` |
| `purify-3.4.16.min.js` | dompurify 3.4.16, `dist/purify.min.js` (MPL 2.0 or Apache 2.0, `LICENSE.dompurify`) | `sha512-sqo+pNp3qRhCIpbgRi1y8Tgk27Bo2Ry7w0dC1NBeNTdZChWjz9Xb/KOoZbRP/R6pQZ80Qw8YhXw13hWWBbMRnQ==` | `sha384-a7SzOxErzJ3ZpQz0zJ32d67dSitNzPcbfybc/ykU9KJhMgZkwqfSxlhhdJRS+XGL` |

To upgrade: download the new tarball, check it against the registry integrity,
copy the same file, and update the file name and both `integrity` attributes in
`static/index.html` and `static/markdown-test.html`.
