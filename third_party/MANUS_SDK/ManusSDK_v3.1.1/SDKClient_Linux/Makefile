CXX      := g++
PYTHON   ?= python3
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
MANUS_SDK_DIR ?= $(THIS_DIR)ManusSDK

#CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror -g --std=c++17
# CXXFLAGS := -g --std=c++17

PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYBIND11_SUFFIX := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
CXXFLAGS := -shared -g --std=c++17 -fPIC ${PYBIND11_INCLUDES}

LDFLAGS  := -L$(MANUS_SDK_DIR)/lib -lManusSDK_Integrated -lncurses -lpthread -Wl,-rpath,$(MANUS_SDK_DIR)/lib
BUILD    := ./
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)

# TARGET   := SDKClient_Linux.out
TARGET   := ManusServer$(PYBIND11_SUFFIX)

INCLUDE  := -I$(MANUS_SDK_DIR)/include
SRC      :=  \
   $(wildcard *.cpp)

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)
	
-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -g
release: all

clean:
	-@rm $(TARGET)
	-@rm -rvf $(OBJ_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"
