CMD_NONE = int(0x00)
CMD_GET_STATUS = int(0x01)

CMD_GET_INPUT_TENSOR = int(0x03)
CMD_SET_INPUT_TENSOR = int(0x83)

CMD_SET_SERVER = int(0x04)
CMD_START_INFER = int(0x84)

CMD_GET_OUTPUT_TENSOR = int(0x05)

CMD_SET_MODEL_PRIMARY = int(0x86)
CMD_SET_MODEL_SECONDARY = int(0x87)
CMD_SET_MODEL_PRIMARY_FLASH = int(0x96)
CMD_SET_MODEL_SECONDARY_FLASH = int(0x97)

CMD_GET_SPEC = int(0x08)

CMD_GET_TIMINGS = int(0x09)

CMD_GET_INPUT_TENSOR_LENGTH = int(0x0A)

CMD_GET_OUTPUT_TENSOR_LENGTH = int(0x0B)

CMD_START_ACQUIRE_SINGLE = int(0x8C)
CMD_START_ACQUIRE_STREAM = int(0x8E)
CMD_START_ACQUIRE_SET_I2C = int(0x8F)

CMD_GET_ACQUIRE_MODE = int(0x0E)

CMD_GET_SENSOR_TENSOR = int(0x0D)
CMD_SET_SENSOR_TENSOR = int(0x8D)

CMD_GET_DEBUG_LOG = int(0x0F)

CMD_GET_ID = int(0x10)

CMD_GET_OUTPUT_GPIO_EN = int(0x11)
CMD_SET_OUTPUT_GPIO_EN = int(0x80 | 0x11)

CMD_GET_OUTPUT_GPIO_THRESH = int(0x12)
CMD_SET_OUTPUT_GPIO_THRESH = int(0x80 | 0x12)

CMD_GET_OUTPUT_GPIO_MODE = int(0x13)
CMD_SET_OUTPUT_GPIO_MODE = int(0x80 | 0x13)

CMD_HELLO = int(0x55)
