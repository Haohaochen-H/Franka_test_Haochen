TABLE_Z_BASE_OFFSET = 0.17

DEFAULT_TABLE_Z_BY_OBJECT = {
    "orange_cube": 0.03,
    "yellow_cube": 0.015,
    "salt_box": 0.015,
}

DEFAULT_OBJECT_HEIGHT_BY_OBJECT = {
    "orange_cube": 0.055,
    "yellow_cube": 0.035,
    "salt_box": 0.055,
}

DEFAULT_PLACE_Z_BY_OBJECT = {
    "salt_box": 0.055,
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


def place_z_for_object(class_name: str, object_id: str = ""):
    for name in (class_name, object_id):
        normalized = normalize_name(name)
        if normalized in DEFAULT_PLACE_Z_BY_OBJECT:
            return DEFAULT_PLACE_Z_BY_OBJECT[normalized]
    return None


def object_height_for_object(class_name: str, object_id: str = ""):
    for name in (class_name, object_id):
        normalized = normalize_name(name)
        if normalized in DEFAULT_OBJECT_HEIGHT_BY_OBJECT:
            return DEFAULT_OBJECT_HEIGHT_BY_OBJECT[normalized]
    table_z = table_z_for_object(class_name=class_name, object_id=object_id)
    return table_z
