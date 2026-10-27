"""Provision dedicated writers only for disposable integration databases."""
from dataclasses import replace
import secrets,uuid
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.storage import Postgres


def configure_writer(service):
    original=service.settings
    assert original.ch_database.startswith(('manager_test_','infra_test_','browser_test_','tushare_test_'))
    admin=ClickHouse(replace(original,ch_writer_mode='disabled'))
    admin.execute(f'CREATE DATABASE IF NOT EXISTS {admin.table("placeholder").split(".")[0]}')
    user='test_writer_'+uuid.uuid4().hex[:12];password=secrets.token_urlsafe(32)
    from betalens_db_manager.security import remember_secrets
    remember_secrets(password)
    import ipaddress
    address=ipaddress.ip_address(admin.execute('SELECT toString(address) FROM system.processes WHERE query_id=currentQueryID() FORMAT TSVRaw'))
    address=getattr(address,'ipv4_mapped',None) or address
    admin.execute(f'CREATE USER `{user}` IDENTIFIED WITH sha256_password BY {admin._literal(password)} HOST IP {admin._literal(str(address))}')
    try:
        admin.execute(f'GRANT SELECT,INSERT,ALTER,CREATE DATABASE,CREATE TABLE,DROP TABLE,TRUNCATE,OPTIMIZE ON `{original.ch_database}`.* TO `{user}`')
        admin.execute(f'GRANT SELECT ON system.* TO `{user}`')
        admin.execute(f'GRANT SHOW ACCESS ON *.* TO `{user}`')
        settings=replace(original,ch_user=user,ch_password=password,ch_writer_mode='local_dedicated')
        service.settings=settings;service.pg=Postgres(settings);service.ch=ClickHouse(settings)
        service.ch.write_coordinator.verify(service.ch)
    except BaseException:
        admin.execute(f'DROP USER IF EXISTS `{user}`')
        admin.execute(f'DROP DATABASE IF EXISTS `{original.ch_database}`');raise
    def cleanup():
        service.ch=admin;service.settings=original;service.pg=Postgres(original)
        admin.execute(f'DROP USER IF EXISTS `{user}`')
    return cleanup
