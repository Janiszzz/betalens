# 逐文件源码索引

由 build_catalog.py 静态生成。返回表达式是源码线索，不是推断出的类型或保证。
docstring 保留作者说明，不意味着已验证正确。无注解/无说明的地方请看正文契约与函数体。

覆盖 **544** 个文件、**2592** 个 Python 类/函数/方法/内部函数声明。
不展开锁文件、研究数据、构建产物；本交接目录本身另在 README 说明。

[返回交接导航](../README.md) · [完整机器可读参数、docstring、返回表达式及调用线索](inventory.json)

## 分组

| 分组 | 文件数 |
| --- | ---: |
| [alpha101_catalog](alpha101_catalog.md) | 404 |
| [build_and_entries](build_and_entries.md) | 9 |
| [core_analyst](core_analyst.md) | 5 |
| [core_backtest](core_backtest.md) | 2 |
| [core_datafeed](core_datafeed.md) | 10 |
| [core_eventstudy](core_eventstudy.md) | 2 |
| [core_exports](core_exports.md) | 1 |
| [core_factor](core_factor.md) | 11 |
| [core_robust](core_robust.md) | 3 |
| [dashboard_backend](dashboard_backend.md) | 8 |
| [dashboard_frontend](dashboard_frontend.md) | 11 |
| [db_manager](db_manager.md) | 28 |
| [db_migrations](db_migrations.md) | 11 |
| [factor_catalog_and_templates](factor_catalog_and_templates.md) | 27 |
| [research_course](research_course.md) | 5 |
| [tests](tests.md) | 7 |

## 全文件快速定位

| 文件 | 作用 |
| --- | --- |
| [.github/workflows/publish.yml](build_and_entries.md#file-4639989a2b9c) | 发布 release 后构建并发布 PyPI |
| [.readthedocs.yaml](build_and_entries.md#file-8b28ba38a2c1) | 运行/构建声明式配置 |
| [betalens-factor/LiqDemand/DISP/factor_DISP.py](factor_catalog_and_templates.md#file-57d1cb3b3ec8) | DISP dispensability factor.；具体因子脚本/模板 |
| [betalens-factor/LiqDemand/DISP/factor_DISP.yaml](factor_catalog_and_templates.md#file-a62d52cd1c9e) | 具体因子的完整运行参数 |
| [betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml](factor_catalog_and_templates.md#file-a07e32b9e773) | 参数空间、搜索与评价规则 |
| [betalens-factor/LiqDemand/DISP/mining/performance.yaml](factor_catalog_and_templates.md#file-0e9c195fdb85) | 挖掘资源、缓存和输出配置 |
| [betalens-factor/LiqDemand/DISP/mining/run.py](factor_catalog_and_templates.md#file-18e6ccaad829) | DISP 挖掘命令入口 |
| [betalens-factor/LiqDemand/class_LiqDemand.yaml](factor_catalog_and_templates.md#file-6ef60084b9db) | 因子类别发现元数据 |
| [betalens-factor/LiqDemand/factor_template_liqdemand.py](factor_catalog_and_templates.md#file-839bf2e7bc21) | 流动性需求因子公共算子和管线 |
| [betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py](alpha101_catalog.md#file-f4929c92773c) | ALPHA1 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml](alpha101_catalog.md#file-e0733fbb1843) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py](alpha101_catalog.md#file-6fe90f79ce3e) | ALPHA1 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml](alpha101_catalog.md#file-42de59f6afe0) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py](alpha101_catalog.md#file-d05cc06824ba) | ALPHA10 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml](alpha101_catalog.md#file-5b1b630312ff) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py](alpha101_catalog.md#file-3c8a70e1f574) | ALPHA10 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml](alpha101_catalog.md#file-afe80ea5ab94) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py](alpha101_catalog.md#file-fa34f8de8078) | ALPHA100 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml](alpha101_catalog.md#file-4fcd634acf3c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py](alpha101_catalog.md#file-2285bf651ec7) | ALPHA100 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml](alpha101_catalog.md#file-2a004b8fdbd3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py](alpha101_catalog.md#file-d32ff6b550f0) | ALPHA101 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml](alpha101_catalog.md#file-778ba29b8487) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py](alpha101_catalog.md#file-4381aac3b9f1) | ALPHA101 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml](alpha101_catalog.md#file-04155b5e913d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py](alpha101_catalog.md#file-5732d95dd387) | ALPHA11 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml](alpha101_catalog.md#file-c8bf5ea6675d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py](alpha101_catalog.md#file-e5f1d4a31f5a) | ALPHA11 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml](alpha101_catalog.md#file-65877ddcabb4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py](alpha101_catalog.md#file-da112b69ef45) | ALPHA12 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml](alpha101_catalog.md#file-0af5da4b93d8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py](alpha101_catalog.md#file-0e3b800cf4eb) | ALPHA12 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml](alpha101_catalog.md#file-d846392587fe) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py](alpha101_catalog.md#file-2e3499323c94) | ALPHA13 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml](alpha101_catalog.md#file-ed77e977030f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py](alpha101_catalog.md#file-bcb99a4f1979) | ALPHA13 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml](alpha101_catalog.md#file-00b52dffb2b9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py](alpha101_catalog.md#file-2b7f9cbb9a9d) | ALPHA14 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml](alpha101_catalog.md#file-d26b5f5de57e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py](alpha101_catalog.md#file-86f10aef9e91) | ALPHA14 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml](alpha101_catalog.md#file-b5fa221e0c69) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py](alpha101_catalog.md#file-252d70cb8682) | ALPHA15 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml](alpha101_catalog.md#file-dd3fae2c5754) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py](alpha101_catalog.md#file-4451b9f8f22f) | ALPHA15 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml](alpha101_catalog.md#file-ed61ac7be679) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py](alpha101_catalog.md#file-1ed0730d0831) | ALPHA16 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml](alpha101_catalog.md#file-2ce68da052d8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py](alpha101_catalog.md#file-89d751b2833c) | ALPHA16 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml](alpha101_catalog.md#file-64d3d9556bf8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py](alpha101_catalog.md#file-ec2790215241) | ALPHA17 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml](alpha101_catalog.md#file-7e56e9c01c4c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py](alpha101_catalog.md#file-7f0fe9196f3f) | ALPHA17 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml](alpha101_catalog.md#file-f283c26834f9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py](alpha101_catalog.md#file-6342d0c44fb2) | ALPHA18 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml](alpha101_catalog.md#file-1ea4011c851a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py](alpha101_catalog.md#file-e1a5b3610af5) | ALPHA18 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml](alpha101_catalog.md#file-02448e5a4a39) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py](alpha101_catalog.md#file-e9a5232cc2b0) | ALPHA19 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml](alpha101_catalog.md#file-070408140899) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py](alpha101_catalog.md#file-14d68c0b9085) | ALPHA19 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml](alpha101_catalog.md#file-a5a032324bd9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py](alpha101_catalog.md#file-f6f87b634a57) | ALPHA2 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml](alpha101_catalog.md#file-1f9f5afac6f0) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py](alpha101_catalog.md#file-f6c8d18db801) | ALPHA2 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml](alpha101_catalog.md#file-d9cb9b2c4dc4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py](alpha101_catalog.md#file-988ddbece890) | ALPHA20 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml](alpha101_catalog.md#file-3c6943908700) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py](alpha101_catalog.md#file-753ec61331c7) | ALPHA20 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml](alpha101_catalog.md#file-a2e17407b35b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py](alpha101_catalog.md#file-3fc784995732) | ALPHA21 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml](alpha101_catalog.md#file-1ed0c57f5f7a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py](alpha101_catalog.md#file-dfae8a64b493) | ALPHA21 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml](alpha101_catalog.md#file-86ad7c4eef6c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py](alpha101_catalog.md#file-f57c2f97fb09) | ALPHA22 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml](alpha101_catalog.md#file-55d9b0986987) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py](alpha101_catalog.md#file-e536345f4a24) | ALPHA22 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml](alpha101_catalog.md#file-fe2089dd0e62) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py](alpha101_catalog.md#file-d1c555a8fe59) | ALPHA23 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml](alpha101_catalog.md#file-ef9046d1f285) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py](alpha101_catalog.md#file-ce988da80287) | ALPHA23 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml](alpha101_catalog.md#file-cc4df2ed6054) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py](alpha101_catalog.md#file-23cced0ea987) | ALPHA24 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml](alpha101_catalog.md#file-adf7d77a3b36) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py](alpha101_catalog.md#file-82c2c6232dea) | ALPHA24 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml](alpha101_catalog.md#file-3e887a2ffda3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py](alpha101_catalog.md#file-e6eafbd16628) | ALPHA25 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml](alpha101_catalog.md#file-3afb69155a77) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py](alpha101_catalog.md#file-e0e099d4dc89) | ALPHA25 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml](alpha101_catalog.md#file-8eb9b386080f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py](alpha101_catalog.md#file-368258e9ecbb) | ALPHA26 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml](alpha101_catalog.md#file-160a5deec2e2) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py](alpha101_catalog.md#file-dc272d147357) | ALPHA26 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml](alpha101_catalog.md#file-da884a19081c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py](alpha101_catalog.md#file-1fa0537737dd) | ALPHA27 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml](alpha101_catalog.md#file-a253a8b7bf9d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py](alpha101_catalog.md#file-9d74b5a7d51d) | ALPHA27 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml](alpha101_catalog.md#file-dab0bfcf6aa9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py](alpha101_catalog.md#file-7e39f0536a02) | ALPHA28 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml](alpha101_catalog.md#file-f0814fab5d3b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py](alpha101_catalog.md#file-d9d4d0cc9083) | ALPHA28 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml](alpha101_catalog.md#file-c639db84c83d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py](alpha101_catalog.md#file-66bdaaf320d5) | ALPHA29 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml](alpha101_catalog.md#file-d5c13025038f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py](alpha101_catalog.md#file-fa2814641367) | ALPHA29 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml](alpha101_catalog.md#file-ecadf5ba956f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py](alpha101_catalog.md#file-49571cbe4ea9) | ALPHA3 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml](alpha101_catalog.md#file-6c706ce36f0b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py](alpha101_catalog.md#file-dc0e49231151) | ALPHA3 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml](alpha101_catalog.md#file-ff24c9e1118f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py](alpha101_catalog.md#file-8bc136955bd9) | ALPHA30 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml](alpha101_catalog.md#file-ebc3f8001419) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py](alpha101_catalog.md#file-5491da0c4c02) | ALPHA30 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml](alpha101_catalog.md#file-d95ff48313f4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py](alpha101_catalog.md#file-461eabe3b7c5) | ALPHA31 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml](alpha101_catalog.md#file-b9c67da27601) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py](alpha101_catalog.md#file-6508b3f8b555) | ALPHA31 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml](alpha101_catalog.md#file-b938bbb9c64e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py](alpha101_catalog.md#file-7a8398bd06fc) | ALPHA32 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml](alpha101_catalog.md#file-0248772f0830) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py](alpha101_catalog.md#file-d596ebf1f6e3) | ALPHA32 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml](alpha101_catalog.md#file-d087bc19db4c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py](alpha101_catalog.md#file-e657ba135c0e) | ALPHA33 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml](alpha101_catalog.md#file-cc64f7d85637) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py](alpha101_catalog.md#file-1379f3b75d22) | ALPHA33 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml](alpha101_catalog.md#file-9ee868cbea0c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py](alpha101_catalog.md#file-9a670dfc863c) | ALPHA34 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml](alpha101_catalog.md#file-a71abd6b2456) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py](alpha101_catalog.md#file-810b2b1e8c7a) | ALPHA34 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml](alpha101_catalog.md#file-8df2ae3ebb77) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py](alpha101_catalog.md#file-d6eba0953206) | ALPHA35 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml](alpha101_catalog.md#file-064a4fac375e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py](alpha101_catalog.md#file-08911b982b50) | ALPHA35 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml](alpha101_catalog.md#file-e8cd8f3ae4cb) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py](alpha101_catalog.md#file-bc9f3683fb1e) | ALPHA36 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml](alpha101_catalog.md#file-0809dce0666d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py](alpha101_catalog.md#file-775d10f6666b) | ALPHA36 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml](alpha101_catalog.md#file-fb2d73aa9c10) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py](alpha101_catalog.md#file-923ace5ea4c4) | ALPHA37 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml](alpha101_catalog.md#file-478f8182ba59) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py](alpha101_catalog.md#file-e29289105283) | ALPHA37 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml](alpha101_catalog.md#file-97f5775f05ba) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py](alpha101_catalog.md#file-13d052d1c723) | ALPHA38 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml](alpha101_catalog.md#file-9ea1487131b2) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py](alpha101_catalog.md#file-a6f6f4602bac) | ALPHA38 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml](alpha101_catalog.md#file-175ab607de52) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py](alpha101_catalog.md#file-a5bf02ddc8be) | ALPHA39 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml](alpha101_catalog.md#file-a91c660958cf) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py](alpha101_catalog.md#file-fe4885387889) | ALPHA39 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml](alpha101_catalog.md#file-30acab8aac67) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py](alpha101_catalog.md#file-cd1fce0ce707) | ALPHA4 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml](alpha101_catalog.md#file-01f7585dc31f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py](alpha101_catalog.md#file-c123543ce7fd) | ALPHA4 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml](alpha101_catalog.md#file-28328ae3a6db) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py](alpha101_catalog.md#file-b65f8dba3441) | ALPHA40 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml](alpha101_catalog.md#file-da41fef7709a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py](alpha101_catalog.md#file-213b0d44a9ee) | ALPHA40 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml](alpha101_catalog.md#file-1bb35a8d5053) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py](alpha101_catalog.md#file-7786e88c5d65) | ALPHA41 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml](alpha101_catalog.md#file-e594ec561ae4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py](alpha101_catalog.md#file-b56afed968dc) | ALPHA41 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml](alpha101_catalog.md#file-3d19bdf3f8ac) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py](alpha101_catalog.md#file-90fc57a16e51) | ALPHA42 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml](alpha101_catalog.md#file-b6efd3206fe2) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py](alpha101_catalog.md#file-1941c3f9cd84) | ALPHA42 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml](alpha101_catalog.md#file-3710fad47607) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py](alpha101_catalog.md#file-d86629efe42a) | ALPHA43 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml](alpha101_catalog.md#file-087bbac54183) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py](alpha101_catalog.md#file-6db7425419ac) | ALPHA43 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml](alpha101_catalog.md#file-8603786ea902) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py](alpha101_catalog.md#file-634f35ca4e4d) | ALPHA44 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml](alpha101_catalog.md#file-33608f67a2e8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py](alpha101_catalog.md#file-e0928ec50d28) | ALPHA44 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml](alpha101_catalog.md#file-4cd4464f3d57) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py](alpha101_catalog.md#file-09b7770e2131) | ALPHA45 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml](alpha101_catalog.md#file-998c9d66c9d0) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py](alpha101_catalog.md#file-2fb3856e0492) | ALPHA45 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml](alpha101_catalog.md#file-ab84699883f9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py](alpha101_catalog.md#file-c0900bd93220) | ALPHA46 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml](alpha101_catalog.md#file-2dd0ce060743) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py](alpha101_catalog.md#file-63ea7452a182) | ALPHA46 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml](alpha101_catalog.md#file-c6e141a536f8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py](alpha101_catalog.md#file-449569935b6f) | ALPHA47 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml](alpha101_catalog.md#file-7d89bab8ad1b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py](alpha101_catalog.md#file-670b7ff2c8c9) | ALPHA47 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml](alpha101_catalog.md#file-12eed1e8b8c4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py](alpha101_catalog.md#file-664ee2cbc61b) | ALPHA48 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml](alpha101_catalog.md#file-0e03593f39b3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py](alpha101_catalog.md#file-299ccec905ac) | ALPHA48 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml](alpha101_catalog.md#file-114bcd7b0f4b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py](alpha101_catalog.md#file-799472d428f7) | ALPHA49 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml](alpha101_catalog.md#file-2871249aff9c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py](alpha101_catalog.md#file-aca9c0c19c8f) | ALPHA49 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml](alpha101_catalog.md#file-ab4a9758c78d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py](alpha101_catalog.md#file-cc0967c39654) | ALPHA5 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml](alpha101_catalog.md#file-145b37b8a923) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py](alpha101_catalog.md#file-152f3b5a5d72) | ALPHA5 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml](alpha101_catalog.md#file-dc1f7c53f452) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py](alpha101_catalog.md#file-efaa2d06d8bc) | ALPHA50 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml](alpha101_catalog.md#file-1f922c6bc2aa) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py](alpha101_catalog.md#file-c58ca28a6516) | ALPHA50 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml](alpha101_catalog.md#file-ca2db2115407) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py](alpha101_catalog.md#file-83cbf8948d60) | ALPHA51 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml](alpha101_catalog.md#file-6dd112b0285a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py](alpha101_catalog.md#file-dd2997d4d6ff) | ALPHA51 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml](alpha101_catalog.md#file-ba67a718fd28) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py](alpha101_catalog.md#file-8a162cabc615) | ALPHA52 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml](alpha101_catalog.md#file-4e8596e08075) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py](alpha101_catalog.md#file-f29e9df04147) | ALPHA52 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml](alpha101_catalog.md#file-6db7448e3d13) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py](alpha101_catalog.md#file-0564af53a3f4) | ALPHA53 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml](alpha101_catalog.md#file-df51092fd9d8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py](alpha101_catalog.md#file-cb55b809eb99) | ALPHA53 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml](alpha101_catalog.md#file-446b51e3f2c3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py](alpha101_catalog.md#file-347f9e051702) | ALPHA54 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml](alpha101_catalog.md#file-3419b51df918) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py](alpha101_catalog.md#file-a23674b955cb) | ALPHA54 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml](alpha101_catalog.md#file-f5a2bd41accb) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py](alpha101_catalog.md#file-0cc0680a7ee4) | ALPHA55 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml](alpha101_catalog.md#file-f1448c8f756b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py](alpha101_catalog.md#file-5fa0f5642f88) | ALPHA55 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml](alpha101_catalog.md#file-b4199452b94b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py](alpha101_catalog.md#file-38e192392092) | ALPHA56 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml](alpha101_catalog.md#file-6ca83f818843) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py](alpha101_catalog.md#file-c957ad0dfa9c) | ALPHA56 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml](alpha101_catalog.md#file-2b20db131fb1) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py](alpha101_catalog.md#file-15af0eed3936) | ALPHA57 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml](alpha101_catalog.md#file-37c859ddf13b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py](alpha101_catalog.md#file-b5fce03da44c) | ALPHA57 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml](alpha101_catalog.md#file-c1426ccb8eae) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py](alpha101_catalog.md#file-dec8e3a544ed) | ALPHA58 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml](alpha101_catalog.md#file-3d2d26f9ec8b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py](alpha101_catalog.md#file-52b639bab165) | ALPHA58 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml](alpha101_catalog.md#file-3202f6db6df8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py](alpha101_catalog.md#file-e1869c852327) | ALPHA59 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml](alpha101_catalog.md#file-10053228be6a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py](alpha101_catalog.md#file-0f0c2e781bba) | ALPHA59 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml](alpha101_catalog.md#file-a1d055966d5e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py](alpha101_catalog.md#file-a03db78b700b) | ALPHA6 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml](alpha101_catalog.md#file-9901b402a217) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py](alpha101_catalog.md#file-9a9dde09ac3e) | ALPHA6 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml](alpha101_catalog.md#file-4b5a5066c766) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py](alpha101_catalog.md#file-deb558734a27) | ALPHA60 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml](alpha101_catalog.md#file-98a78563ad59) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py](alpha101_catalog.md#file-a9eab01881cf) | ALPHA60 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml](alpha101_catalog.md#file-8c1bbe9823b8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py](alpha101_catalog.md#file-673af130e35c) | ALPHA61 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml](alpha101_catalog.md#file-96fe8bba0513) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py](alpha101_catalog.md#file-a2acb2220edc) | ALPHA61 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml](alpha101_catalog.md#file-e5ef83fc95d6) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py](alpha101_catalog.md#file-bac83c1fd4c5) | ALPHA62 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml](alpha101_catalog.md#file-1b64f5e963d1) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py](alpha101_catalog.md#file-9f9b65e387d0) | ALPHA62 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml](alpha101_catalog.md#file-383d256bbd9a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py](alpha101_catalog.md#file-b51b47b97063) | ALPHA63 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml](alpha101_catalog.md#file-d0f51e3084a9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py](alpha101_catalog.md#file-2312379b5235) | ALPHA63 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml](alpha101_catalog.md#file-307266b4c94a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py](alpha101_catalog.md#file-b42ae85f857e) | ALPHA64 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml](alpha101_catalog.md#file-76a44bfdd80b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py](alpha101_catalog.md#file-c867b0355bf5) | ALPHA64 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml](alpha101_catalog.md#file-44c0c183dfd9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py](alpha101_catalog.md#file-9351149a4599) | ALPHA65 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml](alpha101_catalog.md#file-7cac2776969b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py](alpha101_catalog.md#file-29ddbe6fa5f8) | ALPHA65 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml](alpha101_catalog.md#file-77d674dc1eef) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py](alpha101_catalog.md#file-315b9c4af69a) | ALPHA66 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml](alpha101_catalog.md#file-2bfdc122c312) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py](alpha101_catalog.md#file-526203dbded7) | ALPHA66 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml](alpha101_catalog.md#file-26cadafeb738) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py](alpha101_catalog.md#file-6b649f8cdccb) | ALPHA67 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml](alpha101_catalog.md#file-8c935eaace15) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py](alpha101_catalog.md#file-a2324336596d) | ALPHA67 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml](alpha101_catalog.md#file-e3a88bc835b3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py](alpha101_catalog.md#file-841ac1e4a455) | ALPHA68 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml](alpha101_catalog.md#file-71f3e9182843) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py](alpha101_catalog.md#file-151bfbfb0d7e) | ALPHA68 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml](alpha101_catalog.md#file-b20ce11371fd) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py](alpha101_catalog.md#file-9728e8e4de12) | ALPHA69 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml](alpha101_catalog.md#file-accb560cb1a2) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py](alpha101_catalog.md#file-ab7a1cc23e25) | ALPHA69 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml](alpha101_catalog.md#file-cb1b72e6d473) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py](alpha101_catalog.md#file-3e958223e09a) | ALPHA7 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml](alpha101_catalog.md#file-ae385ee28331) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py](alpha101_catalog.md#file-5167e1851771) | ALPHA7 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml](alpha101_catalog.md#file-062d276e63bb) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py](alpha101_catalog.md#file-14c9e98fe932) | ALPHA70 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml](alpha101_catalog.md#file-9b8f9ff81356) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py](alpha101_catalog.md#file-ecd699dee503) | ALPHA70 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml](alpha101_catalog.md#file-e85c97ecdb7e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py](alpha101_catalog.md#file-d1d5c4a24783) | ALPHA71 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml](alpha101_catalog.md#file-69dc8b979812) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py](alpha101_catalog.md#file-6b343d0ff3f1) | ALPHA71 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml](alpha101_catalog.md#file-1161e66d4ca1) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py](alpha101_catalog.md#file-48ffdc7ca61a) | ALPHA72 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml](alpha101_catalog.md#file-622469e61e82) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py](alpha101_catalog.md#file-1c10b4dcb35a) | ALPHA72 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml](alpha101_catalog.md#file-d5b02e5ac516) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py](alpha101_catalog.md#file-a21afbfa9dd6) | ALPHA73 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml](alpha101_catalog.md#file-effc7266c6aa) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py](alpha101_catalog.md#file-a49201d6bf05) | ALPHA73 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml](alpha101_catalog.md#file-d78ec1043ac0) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py](alpha101_catalog.md#file-386b42ec5d57) | ALPHA74 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml](alpha101_catalog.md#file-7e6643bee937) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py](alpha101_catalog.md#file-627fc533a1da) | ALPHA74 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml](alpha101_catalog.md#file-774c8ea64a9e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py](alpha101_catalog.md#file-db04a53f3cad) | ALPHA75 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml](alpha101_catalog.md#file-591560b54b13) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py](alpha101_catalog.md#file-5ae2a0a786cf) | ALPHA75 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml](alpha101_catalog.md#file-3c7573ecfd01) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py](alpha101_catalog.md#file-273eccb0d1e4) | ALPHA76 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml](alpha101_catalog.md#file-0117f0fda6b3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py](alpha101_catalog.md#file-1b146506922b) | ALPHA76 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml](alpha101_catalog.md#file-6fd874ca0ea4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py](alpha101_catalog.md#file-b0f91d1709b4) | ALPHA77 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml](alpha101_catalog.md#file-5719b68e3aff) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py](alpha101_catalog.md#file-d5ab0ce0e90c) | ALPHA77 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml](alpha101_catalog.md#file-84d4892a0d0e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py](alpha101_catalog.md#file-6a9250151302) | ALPHA78 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml](alpha101_catalog.md#file-6618085808d7) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py](alpha101_catalog.md#file-2fa6c62ff896) | ALPHA78 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml](alpha101_catalog.md#file-5e61783ed561) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py](alpha101_catalog.md#file-28bea6ce407e) | ALPHA79 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml](alpha101_catalog.md#file-51f9105ebdf2) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py](alpha101_catalog.md#file-6c31b23caf92) | ALPHA79 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml](alpha101_catalog.md#file-6e4fa6f87f84) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py](alpha101_catalog.md#file-3492ae9e3dcf) | ALPHA8 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml](alpha101_catalog.md#file-c80eeff31e5f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py](alpha101_catalog.md#file-8829674d4a84) | ALPHA8 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml](alpha101_catalog.md#file-96f97a4bc31b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py](alpha101_catalog.md#file-07c7e0eea7d7) | ALPHA80 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml](alpha101_catalog.md#file-9381b95598f6) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py](alpha101_catalog.md#file-0add03c362a6) | ALPHA80 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml](alpha101_catalog.md#file-d177a879b75d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py](alpha101_catalog.md#file-eea4770e1e15) | ALPHA81 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml](alpha101_catalog.md#file-8e9ed858c298) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py](alpha101_catalog.md#file-385ffe3cfd7a) | ALPHA81 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml](alpha101_catalog.md#file-cd2009eded60) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py](alpha101_catalog.md#file-56b338525abb) | ALPHA82 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml](alpha101_catalog.md#file-76419b58262b) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py](alpha101_catalog.md#file-32b049a48b64) | ALPHA82 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml](alpha101_catalog.md#file-7ff385eba426) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py](alpha101_catalog.md#file-de0d764dacb8) | ALPHA83 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml](alpha101_catalog.md#file-f1fb6218b8c6) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py](alpha101_catalog.md#file-2a85a6243231) | ALPHA83 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml](alpha101_catalog.md#file-ca96eba3d79f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py](alpha101_catalog.md#file-00803e89af77) | ALPHA84 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml](alpha101_catalog.md#file-4c320e79ac0f) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py](alpha101_catalog.md#file-02786166ebc8) | ALPHA84 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml](alpha101_catalog.md#file-2d0f7dd5dfc3) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py](alpha101_catalog.md#file-99752000e5e1) | ALPHA85 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml](alpha101_catalog.md#file-c093a2a58e9d) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py](alpha101_catalog.md#file-38810f88571d) | ALPHA85 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml](alpha101_catalog.md#file-e9355840bf8e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py](alpha101_catalog.md#file-4d22420e12b5) | ALPHA86 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml](alpha101_catalog.md#file-5070dcb0d673) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py](alpha101_catalog.md#file-e84e77ed5e3b) | ALPHA86 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml](alpha101_catalog.md#file-34c66a30d60e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py](alpha101_catalog.md#file-75cafd9d21db) | ALPHA87 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml](alpha101_catalog.md#file-ca6eb91986ba) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py](alpha101_catalog.md#file-fc1c5e94a59f) | ALPHA87 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml](alpha101_catalog.md#file-2596e31c1a79) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py](alpha101_catalog.md#file-84359d8f3eb7) | ALPHA88 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml](alpha101_catalog.md#file-5bd1f20164d0) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py](alpha101_catalog.md#file-90dec0a28311) | ALPHA88 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml](alpha101_catalog.md#file-1821814f64e4) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py](alpha101_catalog.md#file-60a529d302ab) | ALPHA89 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml](alpha101_catalog.md#file-ae4e29bc89aa) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py](alpha101_catalog.md#file-b60c2f5c9455) | ALPHA89 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml](alpha101_catalog.md#file-af5e6e0c20b6) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py](alpha101_catalog.md#file-1edf69094b89) | ALPHA9 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml](alpha101_catalog.md#file-d55dd9007d3c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py](alpha101_catalog.md#file-2a4817cfa222) | ALPHA9 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml](alpha101_catalog.md#file-551f6fd52c7a) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py](alpha101_catalog.md#file-3283d2fcf9bc) | ALPHA90 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml](alpha101_catalog.md#file-35132c4881a7) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py](alpha101_catalog.md#file-6b9b9041e7c8) | ALPHA90 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml](alpha101_catalog.md#file-db4419712897) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py](alpha101_catalog.md#file-d3cbfe2317e5) | ALPHA91 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml](alpha101_catalog.md#file-99afce5c7680) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py](alpha101_catalog.md#file-379211040955) | ALPHA91 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml](alpha101_catalog.md#file-170e60915958) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py](alpha101_catalog.md#file-8ca21a96a32f) | ALPHA92 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml](alpha101_catalog.md#file-0e9c204682e7) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py](alpha101_catalog.md#file-0eee0089b6b3) | ALPHA92 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml](alpha101_catalog.md#file-f754034a0b6c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py](alpha101_catalog.md#file-ff95bf4d0c7c) | ALPHA93 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml](alpha101_catalog.md#file-7bb126539f35) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py](alpha101_catalog.md#file-8043387c0d74) | ALPHA93 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml](alpha101_catalog.md#file-da573a99676e) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py](alpha101_catalog.md#file-ac90a8d6b18f) | ALPHA94 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml](alpha101_catalog.md#file-9b249c3209b7) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py](alpha101_catalog.md#file-163e512cce52) | ALPHA94 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml](alpha101_catalog.md#file-2702c045bae8) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py](alpha101_catalog.md#file-8622a8a93942) | ALPHA95 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml](alpha101_catalog.md#file-514dda6fba18) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py](alpha101_catalog.md#file-b71a3818ccc1) | ALPHA95 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml](alpha101_catalog.md#file-d0a1b17cf0a5) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py](alpha101_catalog.md#file-66137b094b81) | ALPHA96 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml](alpha101_catalog.md#file-778541ee8268) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py](alpha101_catalog.md#file-246db38fe38f) | ALPHA96 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml](alpha101_catalog.md#file-dd0f795c9ef9) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py](alpha101_catalog.md#file-fe23d6d56213) | ALPHA97 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml](alpha101_catalog.md#file-0385ba97140c) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py](alpha101_catalog.md#file-fa8a9be2ecd1) | ALPHA97 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml](alpha101_catalog.md#file-d8462301a0d5) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py](alpha101_catalog.md#file-d3a3ffc98dc5) | ALPHA98 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml](alpha101_catalog.md#file-34db958d3244) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py](alpha101_catalog.md#file-b652fad34312) | ALPHA98 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml](alpha101_catalog.md#file-5b0d72c60117) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py](alpha101_catalog.md#file-e94a52062d43) | ALPHA99 cross-sectional factor.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml](alpha101_catalog.md#file-b1ac751e6119) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py](alpha101_catalog.md#file-55e511090dae) | ALPHA99 single-stock timing strategy.；具体因子脚本/模板 |
| [betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml](alpha101_catalog.md#file-67fd7ae1b536) | 具体因子的完整运行参数 |
| [betalens-factor/alpha101/alpha101_formulas.py](factor_catalog_and_templates.md#file-79dca30ba712) | 101 个公式、依赖与回看需求注册 |
| [betalens-factor/alpha101/alpha101_mining.py](factor_catalog_and_templates.md#file-c331801946dd) | Alpha101 的挖掘 hook |
| [betalens-factor/alpha101/alpha101_parameters.py](factor_catalog_and_templates.md#file-fcaf4f4ce126) | Alpha 参数空间推导 |
| [betalens-factor/alpha101/class_alpha101.yaml](factor_catalog_and_templates.md#file-768c6870f501) | 因子类别发现元数据 |
| [betalens-factor/alpha101/factor_template_alpha101.py](factor_catalog_and_templates.md#file-02b6c181729c) | Alpha 算子与截面/择时模板桥接 |
| [betalens-factor/alpha101/mining/parameter_space.yaml](factor_catalog_and_templates.md#file-d9cd32d64d2e) | 参数空间、搜索与评价规则 |
| [betalens-factor/alpha101/mining/performance.yaml](factor_catalog_and_templates.md#file-7663d0e0162f) | 挖掘资源、缓存和输出配置 |
| [betalens-factor/alpha101/mining/run.py](factor_catalog_and_templates.md#file-345c17407b66) | Alpha101 挖掘命令入口 |
| [betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py](factor_catalog_and_templates.md#file-f13bf768b75e) | ILLIQ_v2 factor.；具体因子脚本/模板 |
| [betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml](factor_catalog_and_templates.md#file-7273963b79c7) | 具体因子的完整运行参数 |
| [betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml](factor_catalog_and_templates.md#file-7122ee51949f) | 因子类别发现元数据 |
| [betalens-factor/factor_template.py](factor_catalog_and_templates.md#file-8fbde51b40fc) | 通用截面研究管线与中间数据转换 |
| [betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py](factor_catalog_and_templates.md#file-7d5ea8e8fc1a) | RSI_FAST tdx factor.；具体因子脚本/模板 |
| [betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml](factor_catalog_and_templates.md#file-4bd031c517bb) | 具体因子的完整运行参数 |
| [betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py](factor_catalog_and_templates.md#file-c703dbd74726) | RSI_SLOW tdx factor.；具体因子脚本/模板 |
| [betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml](factor_catalog_and_templates.md#file-46cdfb650f79) | 具体因子的完整运行参数 |
| [betalens-factor/tdx/class_tdx.yaml](factor_catalog_and_templates.md#file-6a9c53d40024) | 因子类别发现元数据 |
| [betalens-factor/tdx/factor_template_tdx.py](factor_catalog_and_templates.md#file-6088c1e0f187) | 通达信公式算子及管线适配 |
| [betalens-factor/tools/eventstudy/eventstudy.yaml](factor_catalog_and_templates.md#file-bcc6e681028c) | 运行/构建声明式配置 |
| [betalens-factor/tools/eventstudy/run_eventstudy.py](factor_catalog_and_templates.md#file-44dfa050b053) | 事件研究 CLI |
| [betalens/__init__.py](core_exports.md#file-7ae211babe9c) | 公共包入口和部分延迟导出 |
| [betalens/analyst/__init__.py](core_analyst.md#file-53b913cb1ee5) | Analyst模块 - 策略评价与报告工具 |
| [betalens/analyst/analyst.py](core_analyst.md#file-4d1bd46c5884) | 评价门面及报告导出 |
| [betalens/analyst/metrics.py](core_analyst.md#file-841e0ef10030) | 纯绩效指标和收益明细计算 |
| [betalens/analyst/naming.py](core_analyst.md#file-1cc403d0bbed) | 证券代码名称映射和缓存 |
| [betalens/analyst/plotting.py](core_analyst.md#file-0e3196c1cb60) | 静态和交互图形 |
| [betalens/backtest/__init__.py](core_backtest.md#file-b96113b139e4) | Backtest模块 - 回测功能 |
| [betalens/backtest/backtest.py](core_backtest.md#file-c07be2fe05e7) | 权重到实际持仓、损益与净值 |
| [betalens/datafeed/__init__.py](core_datafeed.md#file-3f481da29674) | Lightweight datafeed package exports used by the factor pipeline. |
| [betalens/datafeed/config.example.json](core_datafeed.md#file-a22b7673f6ea) | 本地连接配置模板 |
| [betalens/datafeed/config.py](core_datafeed.md#file-d7a892129572) | 分层配置和缓存 |
| [betalens/datafeed/core.py](core_datafeed.md#file-300038c0ad6b) | 研究取数门面与交易日历 |
| [betalens/datafeed/industry.py](core_datafeed.md#file-bf00e03912b5) | 历史行业归属查询 |
| [betalens/datafeed/pool.py](core_datafeed.md#file-4c0ff3abab41) | 线程安全只读连接池 |
| [betalens/datafeed/query.py](core_datafeed.md#file-e3fa2d70263c) | 参数化查询与新旧结构路由 |
| [betalens/datafeed/registry.py](core_datafeed.md#file-ff958942dfcb) | 读层逻辑数据集与核心指标映射 |
| [betalens/datafeed/universe.py](core_datafeed.md#file-ad80ef4db74f) | 历史指数成分查询 |
| [betalens/datafeed/validation.py](core_datafeed.md#file-abc11d466ee2) | 数据缺失、日期、重复检查修复 |
| [betalens/eventstudy/__init__.py](core_eventstudy.md#file-e01684539837) | 包导出/包标识 |
| [betalens/eventstudy/eventstudy.py](core_eventstudy.md#file-92a9334f6d00) | 标准交易日事件窗口研究 |
| [betalens/factor/__init__.py](core_factor.md#file-e157395e4dc9) | 包导出/包标识 |
| [betalens/factor/config.py](core_factor.md#file-fc0b9eb70b7f) | 完整 YAML 校验和参数映射 |
| [betalens/factor/factor.py](core_factor.md#file-a7d119e96fb0) | 可交易池、单/双/多特征分组与权重 |
| [betalens/factor/mining.py](core_factor.md#file-65e954bccf78) | 参数挖掘调度和窗口评价 |
| [betalens/factor/mining_audit.py](core_factor.md#file-6984acfade1d) | 搜索结果持久化与人可读审计 |
| [betalens/factor/mining_cache.py](core_factor.md#file-376c527978b2) | 不可变行情缓存发布和切片 |
| [betalens/factor/mining_optuna.py](core_factor.md#file-7cc8c5ed809b) | 采样器、范围扩展与扰动适配 |
| [betalens/factor/preprocessing.py](core_factor.md#file-949c3892d2a8) | 去极值、标准化与中性化 |
| [betalens/factor/profiling.py](core_factor.md#file-14d5b7575b99) | 因子值体检与跨因子比较 |
| [betalens/factor/signal.py](core_factor.md#file-60bd554734a5) | 择时信号到目标权重转换 |
| [betalens/factor/stats.py](core_factor.md#file-96ca2a399d41) | IC、截面与择时统计及图形 |
| [betalens/robust/__init__.py](core_robust.md#file-82993d6b018a) | 包导出/包标识 |
| [betalens/robust/newrobust.py](core_robust.md#file-cd7713f874f1) | 另一版 Lucky Factors 实现 |
| [betalens/robust/robust.py](core_robust.md#file-1e086893dced) | 当前公开导出的 RobustTest 与旧辅助工具 |
| [betalens_db_manager/__init__.py](db_manager.md#file-897bd0749fe2) | Local database management tools for Betalens. |
| [betalens_db_manager/__main__.py](db_manager.md#file-de4611a5aeb9) | CLI/桌面启动分发 |
| [betalens_db_manager/adapters/__init__.py](db_manager.md#file-baeb0a7a316d) | Source adapters owned by :mod:'betalens_db_manager'. |
| [betalens_db_manager/adapters/ede.py](db_manager.md#file-d8838edeaf5f) | Wind EDE 宽表解析 |
| [betalens_db_manager/adapters/files.py](db_manager.md#file-637a53b269ae) | CSV/Excel 读取、分块与时间对齐 |
| [betalens_db_manager/adapters/industry.py](db_manager.md#file-9c8c070304c1) | 行业源数据规范化 |
| [betalens_db_manager/adapters/wind.py](db_manager.md#file-945a99b92993) | 可选 WindPy 行情获取 |
| [betalens_db_manager/constants.py](db_manager.md#file-41d76e9300ae) | 导入列、模式、日志路径等常量 |
| [betalens_db_manager/contracts.py](db_manager.md#file-4863b03ad0dd) | 每版结构契约 |
| [betalens_db_manager/db.py](db_manager.md#file-f91cbc937870) | 管理端连接和分页查询 |
| [betalens_db_manager/gui.py](db_manager.md#file-512c49ec65e0) | GUI 公共兼容启动入口 |
| [betalens_db_manager/gui_app.py](db_manager.md#file-a68ba68ce6fd) | PySide6 四页桌面界面 |
| [betalens_db_manager/gui_controller.py](db_manager.md#file-035c4ba0651b) | 无 Qt 的界面业务控制 |
| [betalens_db_manager/import_adapters.py](db_manager.md#file-b6247cc64df5) | 带目标类型的源数据适配注册 |
| [betalens_db_manager/import_manifest.example.yaml](db_manager.md#file-0b181dfe56e8) | 运行/构建声明式配置 |
| [betalens_db_manager/import_manifest.py](db_manager.md#file-b0dc95aa2a97) | 多文件清单预检与恢复 |
| [betalens_db_manager/importers.py](db_manager.md#file-dbbdf935cf84) | 旧导入接口兼容与数据库写入器 |
| [betalens_db_manager/init_local.bat](db_manager.md#file-9d1aee965304) | Windows 启动/初始化包装脚本 |
| [betalens_db_manager/job_store.py](db_manager.md#file-c3d0cb04399b) | 本地 SQLite 任务持久化 |
| [betalens_db_manager/jobs.py](db_manager.md#file-eb1cf083fb8f) | 单文件导入任务编排 |
| [betalens_db_manager/manager.py](db_manager.md#file-1c720f6d6eec) | 无 Qt 的数据库服务门面 |
| [betalens_db_manager/migrations/0001_bootstrap.sql](db_migrations.md#file-14788f986bce) | 创建 schema、迁移记录和基础覆盖元数据 |
| [betalens_db_manager/migrations/0002_dimensions.sql](db_migrations.md#file-ab0e4dfec545) | 创建证券、名称历史和指标维度 |
| [betalens_db_manager/migrations/0003_market_fact.sql](db_migrations.md#file-4cfe0839318b) | 创建日行情事实表与指标存储相关结构 |
| [betalens_db_manager/migrations/0004_observation_fact.sql](db_migrations.md#file-f978032b260a) | 创建按可得时间组织的通用观测事实表 |
| [betalens_db_manager/migrations/0005_pit_and_metadata.sql](db_migrations.md#file-0b1ba7c894e5) | 创建行业、指数成分、交易状态等 PIT 结构 |
| [betalens_db_manager/migrations/0006_migrate_legacy.sql](db_migrations.md#file-abf4b21cf55b) | 将旧结构长表迁移进规范化结构 |
| [betalens_db_manager/migrations/0007_compatibility_views.sql](db_migrations.md#file-afaa44f571f5) | 保存旧关系并建立兼容视图 |
| [betalens_db_manager/migrations/0008_finalize.sql](db_migrations.md#file-c16ab4922b07) | 初始化覆盖信息并完成迁移收尾 |
| [betalens_db_manager/migrations/0009_lifecycle_audit.sql](db_migrations.md#file-1171d4c99f63) | 完善生命周期、名称历史及审计相关结构 |
| [betalens_db_manager/migrations/0010_trade_calendar.sql](db_migrations.md#file-6e49529b11ba) | 增加按交易所和日期存储的交易日历 |
| [betalens_db_manager/migrations/__init__.py](db_migrations.md#file-2f12ab448721) | Packaged, immutable SQL migrations for :mod:'betalens_db_manager'. |
| [betalens_db_manager/profiles.py](db_manager.md#file-d37e0ef025d5) | 连接档案选择与来源解释 |
| [betalens_db_manager/records.py](db_manager.md#file-ed9869060643) | 导入记录兼容门面 |
| [betalens_db_manager/registry.py](db_manager.md#file-5811a0034afc) | 管理端逻辑数据集、指标与可写性 |
| [betalens_db_manager/run.bat](db_manager.md#file-f04d8cd69b37) | Windows 启动/初始化包装脚本 |
| [betalens_db_manager/schema.py](db_manager.md#file-8fed929085b1) | 版本化建库迁移和核验 |
| [betalens_db_manager/utils.py](db_manager.md#file-ca10f44601c9) | 路径、哈希、JSON 和表格预览工具 |
| [betalens_db_manager/validators.py](db_manager.md#file-6e5a97be098f) | 导入前数据校验 |
| [dashboard/backend/__init__.py](dashboard_backend.md#file-b9cec2a3485f) | FastAPI backend for the rebuilt betalens dashboard. |
| [dashboard/backend/eventstudy_dashboard.py](dashboard_backend.md#file-105f9bea0691) | 事件文件发现与事件结果适配 |
| [dashboard/backend/factors.py](dashboard_backend.md#file-9dd3cb796ae6) | YAML 因子发现、详情与动态加载 |
| [dashboard/backend/main.py](dashboard_backend.md#file-7a2bfea64afa) | HTTP 路由、错误码与日志流 |
| [dashboard/backend/requirements.txt](dashboard_backend.md#file-0387907fe584) | 后端依赖声明 |
| [dashboard/backend/runs.py](dashboard_backend.md#file-80458cab6992) | 内存任务队列与执行状态机 |
| [dashboard/backend/schemas.py](dashboard_backend.md#file-4fb540c77400) | Pydantic 请求响应模型 |
| [dashboard/backend/serialization.py](dashboard_backend.md#file-be7faaf40841) | 研究对象到前端契约 |
| [dashboard/backend/test_eventstudy_dashboard.py](tests.md#file-7a183a1b3480) | 回归测试：test_eventstudy_dashboard.py |
| [dashboard/backend/test_factor_yaml.py](tests.md#file-4a0a2e5b70e5) | 回归测试：test_factor_yaml.py |
| [dashboard/backend/test_serialization.py](tests.md#file-da602679d1fc) | 回归测试：test_serialization.py |
| [dashboard/frontend/package.json](dashboard_frontend.md#file-14cdcf6290e4) | 前端依赖与 npm 脚本 |
| [dashboard/frontend/src/App.tsx](dashboard_frontend.md#file-dc9af101bd32) | 页面、参数表单、轮询日志与结果展示 |
| [dashboard/frontend/src/PlotView.tsx](dashboard_frontend.md#file-ff15510d96a5) | Plotly React 图表包装 |
| [dashboard/frontend/src/api.ts](dashboard_frontend.md#file-4a56ca41d849) | HTTP 客户端、参数编码与重试 |
| [dashboard/frontend/src/main.tsx](dashboard_frontend.md#file-df85405b7a37) | React DOM 挂载 |
| [dashboard/frontend/src/plotly.d.ts](dashboard_frontend.md#file-12d876c67cf0) | Plotly 模块类型声明 |
| [dashboard/frontend/src/styles.css](dashboard_frontend.md#file-93e49b5a5280) | 界面布局和视觉样式 |
| [dashboard/frontend/src/types.ts](dashboard_frontend.md#file-7be4913c0431) | 前端数据契约 |
| [dashboard/frontend/tsconfig.json](dashboard_frontend.md#file-3208b0c1154d) | 前端 TypeScript 编译选项 |
| [dashboard/frontend/tsconfig.node.json](dashboard_frontend.md#file-cbfe98a289bd) | Node/Vite 侧 TypeScript 配置 |
| [dashboard/frontend/vite.config.ts](dashboard_frontend.md#file-ea39566ad183) | Vite、API 代理与后端启动插件 |
| [dashboard/run.bat](build_and_entries.md#file-7ed549aa9f61) | Windows 启动/初始化包装脚本 |
| [dashboard/run_backend.bat](build_and_entries.md#file-0fba722d3629) | Windows 启动/初始化包装脚本 |
| [dashboard/run_frontend.bat](build_and_entries.md#file-66247751d19e) | Windows 启动/初始化包装脚本 |
| [docs/conf.py](build_and_entries.md#file-254a2da7a740) | Sphinx 文档构建配置 |
| [docs/learning/first_factor.py](research_course.md#file-0204328a800a) | 研究员课程练习或模板 |
| [docs/learning/labs.py](research_course.md#file-58f614fdc62d) | 研究员课程练习或模板 |
| [docs/learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py](research_course.md#file-f4c44b805e65) | COURSE_MOM：完整的动量教学因子。市场回测需要真实数据库。；具体因子脚本/模板 |
| [docs/learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml](research_course.md#file-58afd291d184) | 具体因子的完整运行参数 |
| [docs/learning/templates/research/class_research.yaml](research_course.md#file-1fb012006aba) | 因子类别发现元数据 |
| [docs/requirements.txt](build_and_entries.md#file-271b54d579b5) | 文档构建依赖 |
| [pyproject.toml](build_and_entries.md#file-5d07e7d72637) | 包元数据、依赖与构建配置 |
| [requirements.txt](build_and_entries.md#file-19359a61ae24) | 仓库 Python 依赖清单 |
| [tests/test_backtest.py](tests.md#file-97ac7aeb9559) | 回归测试：test_backtest.py |
| [tests/test_eventstudy.py](tests.md#file-58083f919dae) | 回归测试：test_eventstudy.py |
| [tests/test_factor_grouping.py](tests.md#file-f5389f5d4f0a) | 回归测试：test_factor_grouping.py |
| [tests/test_factor_mining.py](tests.md#file-3d0753a5285e) | 回归测试：test_factor_mining.py |
