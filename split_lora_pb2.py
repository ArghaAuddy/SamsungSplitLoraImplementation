# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: Proto/split_lora.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'Proto/split_lora.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16Proto/split_lora.proto\"\'\n\x11\x45mbeddingsRequest\x12\x12\n\nembeddings\x18\x01 \x03(\x02\" \n\x0eLogitsResponse\x12\x0e\n\x06logits\x18\x01 \x03(\x02\x32\x42\n\tSplitLora\x12\x35\n\x0eSendEmbeddings\x12\x12.EmbeddingsRequest\x1a\x0f.LogitsResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'Proto.split_lora_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_EMBEDDINGSREQUEST']._serialized_start=26
  _globals['_EMBEDDINGSREQUEST']._serialized_end=65
  _globals['_LOGITSRESPONSE']._serialized_start=67
  _globals['_LOGITSRESPONSE']._serialized_end=99
  _globals['_SPLITLORA']._serialized_start=101
  _globals['_SPLITLORA']._serialized_end=167
# @@protoc_insertion_point(module_scope)
