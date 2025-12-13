# src/data/__init__.py
"""
Data layer - Database và storage operations.
"""
from .database import (
    init_db,
    get_connection,
    log_attendance,
    get_today_attendance,
    get_attendance_by_date,
    get_summary_by_person,
    get_all_employees,
    add_employee,
    remove_employee,
    get_today_sessions,
    get_active_members,
    export_to_csv,
    get_anomaly_sessions,
    sync_employees_with_face_db,
    midnight_checkout_all_sessions,
)

__all__ = [
    'init_db',
    'get_connection',
    'log_attendance',
    'get_today_attendance',
    'get_attendance_by_date',
    'get_summary_by_person',
    'get_all_employees',
    'add_employee',
    'remove_employee',
    'get_today_sessions',
    'get_active_members',
    'export_to_csv',
    'get_anomaly_sessions',
    'sync_employees_with_face_db',
    'midnight_checkout_all_sessions',
]
