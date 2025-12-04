BUILD_DIR = build
OUTPUT = onnx_ocr

all: clean build run

build:
	cmake -B $(BUILD_DIR) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build $(BUILD_DIR)

run:
	./$(BUILD_DIR)/$(OUTPUT) ./config.yaml

debug:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DIR)

run_debug:
	valgrind -s ./$(BUILD_DIR)/$(OUTPUT) ./config.yaml

clean:
	rm -rf build
