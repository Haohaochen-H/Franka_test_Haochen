TABLE_Z_BASE_OFFSET = 0.17

DEFAULT_TABLE_Z_BY_OBJECT = {
    "orange_cube": 0.03,
    "yellow_cube": 0.02,
}


def target_z_from_table_height(table_z: float) -> float:
    return TABLE_Z_BASE_OFFSET + table_z


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def table_z_for_object(class_name: str, object_id: str = "", override_table_z=None):
    if override_table_z is not None:
        return float(override_table_z)
    for name in (class_name, object_id):
        normalized = normalize_name(name)
        if normalized in DEFAULT_TABLE_Z_BY_OBJECT:
            return DEFAULT_TABLE_Z_BY_OBJECT[normalized]
    return None
