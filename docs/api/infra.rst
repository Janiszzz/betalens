DB Manager API
==============

独立发行包为 ``betalens-db-manager``，Python 包为 ``betalens_db_manager``。
网页和 API 默认位于 ``http://127.0.0.1:8010``，``/docs`` 展示当前服务的 OpenAPI。
流程见 :doc:`../guide/infra`，业务语义见 :doc:`../guide/db-manager-reference`，
本地部署见 :doc:`../guide/local-databases`。

Python 入口
-----------

.. code-block:: python

   from betalens_db_manager import DatabaseManagerService, Settings
   from betalens_db_manager.reference_service import ReferenceWorkflow

   settings = Settings.from_env()
   service = DatabaseManagerService(settings)
   reference = ReferenceWorkflow(service)

``Settings.from_env(env_file=None)`` 不修改全局进程环境。环境文件的选择顺序为：
显式参数、``BETALENS_DB_MANAGER_ENV``、``BETALENS_INFRA_ENV``、默认配置目录。
文件配置覆盖同名进程环境变量。服务构造会初始化本地 YAML 和 SQLite 元数据，
不能当作完全无副作用的配置探测。

包顶层导出 ``Settings/DatabaseManagerService/SchemaManager/SourceRegistry/SubscriptionAdapter/FetchContext``。
旧桌面管理器的 ``DatabaseClient/QueryRequest/ImportJobRunner/ImportRecordStore`` 已退役。

通用本地表
----------

* ``service.registry.save(definition, expected=None)``：保存带系统审计字段的 YAML，不执行 DDL。
* ``service.schema.plan(name, targets=None, action='deploy', destructive=False, rename_to=None)``。
* ``service.schema.apply(token)``：执行确认的结构操作，分别记录两库结果。
* ``service.mappings.document(name)`` / ``save(value, expected)``：读取与校验保存映射。
* ``service.compile(request)``：根据本地表、来源和日期编译请求，不进行网络提数。
* ``service.plan(request)`` / ``download(job)`` / ``preview(job)`` / ``commit(job, preview_token, accept_partial=False)``。
* ``service.resume(job)``：通用任务恢复至待审计；参考任务建立关联新任务，不放行中断版本。
* ``service.audit(job, after=0, limit=100)`` / ``rollback(job, targets=None)``。
* ``service.field_audit(job=None)``：区分映射登记、响应验证与实际存储验证。

通用 HTTP 路由：

* ``GET /api/table-registry``：持久化表清单及实际 PG/CH 部署、行数和日期状态。
* ``POST /api/tables``：保存 ``definition`` 或 ``yaml`` 草稿。
* ``GET /api/tables/{name}/definition``：表定义与文件哈希。
* ``GET|POST /api/tables/{name}/mapping``：映射读取或保存；写入携带 ``file_hash`` 防止覆盖外部编辑。
* ``POST /api/tables/{name}/schema-plan`` / ``/api/tables/schema-apply``：结构预览和确认。
* ``POST /api/tables/archive-backfill``：从已有归档创建补字段任务，不自动入库。
* ``GET|POST /api/sources``：订阅源实例、查询合同及鉴权环境变量引用。
* ``POST /api/tasks/compile`` / ``/api/tasks``：请求匹配、持久化任务。
* ``GET /api/tasks`` / ``/api/tasks/{job}``：任务列表与详情。
* ``POST /api/tasks/{job}/download|preview|commit|resume|cancel|rollback``：登记受管理操作。
* ``GET /api/tasks/{job}/cache|logs|evidence|audit``：缓存分页、日志、转换及入库证据。
* ``POST /api/tasks/{job}/replay``：保留原任务，用当前规则建立关联审计任务。
* ``WS /api/live/{job}``：同源实时事件。
* ``GET|POST /api/config`` / ``POST /api/config/test``：私有配置及连接测试，不回传凭据。
* ``GET /api/write-capabilities``：当前实际可写能力。

网页操作通常返回队列编号；HTTP 200 不代表数据已写入或业务完整。
``/api/jobs`` 与 ``/api/tasks`` 共用旧任务存储。参考任务使用下一节专用路由。

参考表与完整性检查
------------------

``ReferenceWorkflow(service)`` 提供 ``plan(request)``、``download(job)``、
``preview(job)``、``commit(job, token, accept_partial=False)``、
``rollback(job, targets=None)``、``retry(job)``、``detail(job)``、``check(request)``
及 ``baseline(request, source_task_ids)``。
``baseline`` 只复制重新核验通过的原始归档；证据不足标为 ``needs_reacquisition``，
仍需重新审计和确认。服务方法直接执行，网页 POST 执行操作则进入受管理队列。

.. list-table:: 参考表 HTTP 路由
   :header-rows: 1
   :widths: 12 43 45

   * - 方法
     - 路径
     - 请求与结果
   * - POST
     - ``/api/integrity/check``
     - 明确范围检查；返回 ``results[target].rows/summary/complete``。
   * - GET
     - ``/api/reference/tables``
     - 表列、物理键、逻辑键、映射、可信范围及历史行数。
   * - POST
     - ``/api/reference/schema/preview``
     - 输入 ``targets``；返回六张业务/元数据表的 create-only DDL 和 ``token``。
   * - POST
     - ``/api/reference/schema/apply``
     - 输入 ``token``、可选同一 ``targets``；分别创建及检查结构，不迁移旧表。
   * - POST / GET
     - ``/api/reference/tasks``
     - 登记范围请求或列出参考任务；登记不下载。
   * - GET
     - ``/api/reference/tasks/{job}``
     - 请求、尝试、展开的日志列、预览、分库分模块结果及撤销记录。
   * - POST
     - ``/api/reference/tasks/{job}/download`` / ``preview``
     - 排队执行提数并审计，或重新审计缓存。
   * - POST
     - ``/api/reference/tasks/{job}/commit``
     - 输入 ``token``、``accept_partial``；分别追加、核验、发布。
   * - POST
     - ``/api/reference/tasks/{job}/rollback``
     - 输入可选 ``targets``；新任务追加撤销事件，不删除业务行。
   * - POST
     - ``/api/reference/tasks/{job}/retry``
     - 返回关联新任务，尚未开始执行。
   * - POST
     - ``/api/reference/baselines``
     - 输入 ``request`` 和 ``source_task_ids``，准备待审计基线。
   * - GET
     - ``/api/reference/versions?target=postgres&audit=false``
     - 最新可信范围；``audit=true`` 包含未发布、旧版、撤销历史。
   * - GET
     - ``/api/query/{version_table}?target=postgres``
     - 默认只读可信版本，支持 ``code/start/end/limit/after``。

只读检查示例：

.. code-block:: python

   import requests

   response = requests.post(
       "http://127.0.0.1:8010/api/integrity/check",
       json={
           "targets": ["postgres", "clickhouse"],
           "modules": ["calendar", "industry", "suspension"],
           "security_codes": ["000001.SZ"],
           "start_date": "2026-10-12", "end_date": "2026-10-14",
           "exchanges": ["SSE", "SZSE", "BSE"],
           "schemes": ["SW2021", "CITICS"], "levels": [1, 2, 3],
           "check_times": ["10:00:00"],
       }, timeout=30,
   )
   response.raise_for_status()
   result = response.json()["results"]

日期接受 ``YYYY-MM-DD`` 或 ``YYYYMMDD``；股票代码带 ``.SH/.SZ/.BJ``。
行业和停牌请求必须指定股票。``check_times`` 是上海本地时刻，体系值为
``SW2021/SW2014/CITICS``。更新请求可额外携带 ``source_id``、
``requests_per_minute``、``parent_task_id``。
结果状态为 ``complete/unknown/conflict``，``reason`` 区分缺口、失败、待确认、
未发布、中断、撤销、源端无记录、依赖变化及业务边界未知。
任务可信度与业务完整性分开判断，未知不得转成正常交易或休市。

底层 ``ReferenceStore(service, target)`` 位于 ``reference_storage.py``，提供
``install/heads/stage/verify/publish/revoke/snapshot``，日常写入应通过审计确认流程。
``publish`` 核验目标库清单、行数、键、内容哈希及固定的范围版本头；
SQLite 的 ``completed`` 不授予可见性。PG 读取采用可重复读快照；CH 先取发布事件边界，
再读取不可变版本和范围。

多源扩展与精度
--------------

``SourceRegistry`` 支持 HTTP、文件、Tushare；Wind 未配置时不可执行。
新适配器通过 ``betalens_db_manager.sources`` entry point 或 ``register(kind, factory)`` 注册，
继承 ``SubscriptionAdapter``，实现 ``fetch(request, FetchContext)`` 及
``capabilities/plan/validate_response/normalize``。适配器不得直接写业务表。

通用 YAML 映射与固定参考域处理器是两条路径：三个参考域的规则由
``reference_contracts.py/reference_domains.py`` 版本化，编辑通用
``mappings/*.yaml`` 不会改动这些固定业务规则。
HTTP 原文与标准 ``fields/items`` 响应分别归档。
Decimal 和超过 JavaScript 安全范围的整数以字符串传输，避免前端浮点精度损失。
