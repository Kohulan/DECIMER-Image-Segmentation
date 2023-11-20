# Changelog

## [1.3.0](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.5...v1.3.0) (2023-11-20)


### Features

* delete more deprecated mask expansion code ([45e0b97](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/45e0b970f3cf1c0b89421c478bc9a6ce23bd0217))
* delete tests of deleted functions ([99d1b38](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/99d1b3849aba7abe7588d489486f4862ebd158f4))
* new seed pixel determination ([2ef9a5e](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/2ef9a5e7e406407b09e1197f1c0defc2be1250bb))
* remove deprecated expansion code ([1ca862f](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/1ca862f0237134b935d5d32151129d0045057f4c))
* speed-up scikit-image and avoiding loops ([fd874a3](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/fd874a34ec9045b8ab116c34f1821479397e5e10))
* Update demonstration notebook ([e648541](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/e6485410fcebe72ef50b26d4a335d151db42de67))
* update dependencies ([60358c9](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/60358c9f9cc4fd29184d78c84718cb454b8cbfaf))
* Update link to download new model ([28c3397](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/28c33975d1538f6920321f03cc638d38e96f7dd9))


### Bug Fixes

* table line exlusion with connected object detection ([e2f38d7](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/e2f38d7b646fefc39b3401d49fd9a54707ddeb20))
* tests for mask expansion ([3ec8239](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/3ec8239db46b1d92faa89a15e3e1319462f04502))
* update model link README.md ([c5b2cd3](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/c5b2cd3d04133aeaee41ee9fb6a0d263154abb7e))

## [1.2.5](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.4...v1.2.5) (2023-10-19)


### Bug Fixes

* adapt seed pixel determination test according to changes ([337775d](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/337775d34b536ee79aa63e7b0153aae7c49b229d))
* do not return empty segments ([f1deb46](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/f1deb4606627fd96c43ec8527c5287e0823f5905))
* don't use pixels from exclusion mask as seed pixels ([fe96aa9](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/fe96aa91af94c2c8b650ae8d3215a3b497af7469))

## [1.2.4](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.3...v1.2.4) (2023-10-19)


### Bug Fixes

* expansion crash when no structures have been detected [#95](https://github.com/Kohulan/DECIMER-Image-Segmentation/issues/95) ([1242a67](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/1242a676de2ba970d074be151b8b1865b608f8d5))

## [1.2.3](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.2...v1.2.3) (2023-09-21)


### Bug Fixes

* test-commit ([33fd1b7](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/33fd1b70cc021aac0334e29cf85c312ea50fd42f))

## [1.2.2](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.1...v1.2.2) (2023-09-21)


### Bug Fixes

* remove wrong version specification ([b997b43](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/b997b43c7cbb1f581f576a3a6e4fc50fc8aeb2d4))
* stable version specification for ([22b1a51](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/22b1a51a1e90c0b10848cbddbf03a0852548afc3))
* trigger pypi release on release ([7b1eb37](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/7b1eb37efe32473e4268205953aa82d3c3e5518f))

## [1.2.1](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/v1.2.0...v1.2.1) (2023-09-21)


### Bug Fixes

* GH action for pypi releases ([ce4c3b9](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/ce4c3b940d4c9763f6fb352c7cc4bf5895185d2a))

## [1.2.0](https://github.com/Kohulan/DECIMER-Image-Segmentation/compare/1.1.4...v1.2.0) (2023-09-19)


### Features

* adaptive line kernel based on structure size ([2f2fe8e](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/2f2fe8e350b6f88eeda2fa9d2c20967283ef0bb6))
* function to get mean size from bboxes ([70a5b2b](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/70a5b2b66a29dda2e63308e0c24b57093eaefb76))
* GH Action for automated Pypi releases ([dbc55f6](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/dbc55f64ba91899b542ffe5e007ae23ba570d892))


### Bug Fixes

* formatted with black ([b7c7417](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/b7c74173fa62f7f5a880fe85069a7942783d296a))
* linter appeasement ([79c89ca](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/79c89ca4898edca0ad0552ae6681d379198fa426))
* lower relative threshold for kernel for line detection ([1aa4deb](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/1aa4debe8557e8ef379db0b52823e8f5d0dd1ed8))
* max instead of mean structure size for line kernels ([227c452](https://github.com/Kohulan/DECIMER-Image-Segmentation/commit/227c4527cfeb5407daa7b041eb221ff5913420f8))
