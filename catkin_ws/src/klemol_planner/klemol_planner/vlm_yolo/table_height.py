TABLE_Z_BASE_OFFSET = 0.17


def target_z_from_table_height(table_z: float) -> float:
    return TABLE_Z_BASE_OFFSET + table_z
