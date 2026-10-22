DB Manager API
==============

独立分发包 ``betalens-db-manager``，Python 包 ``betalens_db_manager``。

* ``Settings.from_env(env_file=None)``：每次读取私有环境文件，不修改全局进程环境。
* ``DatabaseManagerService(settings)``：本地表、来源、任务、审计和回退的统一入口。
* ``service.registry.save(definition, expected=None)``：保存带系统审计字段的表定义。
* ``service.schema.plan(name, targets=None, action='deploy', destructive=False, rename_to=None)``：生成带令牌的结构操作预览。
* ``service.schema.apply(token)``：执行确认的结构操作，分别记录两库结果。
* ``service.mappings.document(name)`` / ``save(mapping, file_hash)``：图形映射和 YAML 采用同一版本校验。
* ``compile(request)``：由本地表、来源和日期生成请求，尚不访问外部 API。
* ``plan(request)`` / ``download(job)`` / ``preview(job)`` / ``commit(job, token, accept_partial=False)``。
* ``resume(job)``：复用成功归档恢复至待审计状态，入库须再次明确确认。
* ``audit(job, after=0, limit=100)`` / ``rollback(job, targets=None)``：行级证据及有冲突保护的回退。
* ``field_audit(job=None)``：区别映射已登记、响应已验证和实际存储已验证。

``create_app(service=None)`` 提供 REST 和网页，``/docs`` 展示请求参数。

* ``/api/tables`` / ``/api/table-registry``：实际两库状态与持久化表清单。
* ``/api/tables/{name}/definition`` / ``mapping`` / ``coverage``：表定义、映射和时间覆盖。
* ``/api/tables/{name}/schema-plan`` / ``/api/tables/schema-apply``：结构预览和确认。
* ``/api/tables/archive-backfill``：复用已有归档创建补字段任务，确认前不写业务数据。
* ``/api/sources``：订阅源实例及查询字段合同；鉴权只允许环境变量引用。
* ``/api/tasks/compile`` / ``/api/tasks``：请求匹配与持久化任务。
* ``/api/tasks/{job}/download|preview|commit|resume|cancel|rollback``：登记到同一个受管理执行器。
* ``/api/tasks/{job}/cache|logs|evidence|audit``：原始响应、实时日志、字段转换和数据库变更。
* ``/api/tasks/{job}/replay``：保留旧记录，用当前规则建立关联审计任务。
* ``/api/live/{job}``：WebSocket 实时事件，校验同源 Origin。
* ``/api/config`` / ``/api/config/test``：私有配置编辑及连接测试；不回传凭据。

``SourceRegistry`` 支持 HTTP、文件和 Tushare；Wind 未配置时不可执行。新增 Python 适配器可通过
``betalens_db_manager.sources`` entry point 注册，或使用 ``register(kind, factory)``。
新适配器继承 ``SubscriptionAdapter``，实现 ``fetch(request, FetchContext)``，并提供
``capabilities/plan/validate_response/normalize``。适配器负责网络请求和响应合同，公共管理服务负责
分页检查点、标准化、审计和写入。HTTP 原文与标准 fields/items 响应分别归档。

Decimal 和超出 JavaScript 安全范围的整数以字符串传输。结构、映射、归档和目标快照绑定确认令牌；
任何不兼容变化均需重新审计。旧 ``/api/jobs`` 路径用于读取及操作已有任务，与新任务共享持久化存储。
