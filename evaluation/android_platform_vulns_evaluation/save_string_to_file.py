import os

# Input string (you can also replace this with input() or argparse if needed)
input_string = """"diff --git a/system/stack/gatt/gatt_sr.cc b/system/stack/gatt/gatt_sr.cc\nindex 6f81b90514..c61df52bc8 100644\n--- a/system/stack/gatt/gatt_sr.cc\n+++ b/system/stack/gatt/gatt_sr.cc\n@@ -734,6 +734,11 @@ void gatts_process_primary_service_req(tGATT_TCB& tcb, uint16_t cid,\n \n   uint16_t payload_size = gatt_tcb_get_payload_size_tx(tcb, cid);\n \n+  // This can happen if the channel is already closed.\n+  if (payload_size == 0) {\n+    return;\n+  }\n+\n   uint16_t msg_len =\n       (uint16_t)(sizeof(BT_HDR) + payload_size + L2CAP_MIN_OFFSET);\n   BT_HDR* p_msg = (BT_HDR*)osi_calloc(msg_len);\n@@ -769,6 +774,12 @@ static void gatts_process_find_info(tGATT_TCB& tcb, uint16_t cid,\n   }\n \n   uint16_t payload_size = gatt_tcb_get_payload_size_tx(tcb, cid);\n+\n+  // This can happen if the channel is already closed.\n+  if (payload_size == 0) {\n+    return;\n+  }\n+\n   uint16_t buf_len =\n       (uint16_t)(sizeof(BT_HDR) + payload_size + L2CAP_MIN_OFFSET);\n \n@@ -902,6 +913,11 @@ static void gatts_process_read_by_type_req(tGATT_TCB& tcb, uint16_t cid,\n \n   uint16_t payload_size = gatt_tcb_get_payload_size_tx(tcb, cid);\n \n+  // This can happen if the channel is already closed.\n+  if (payload_size == 0) {\n+    return;\n+  }\n+\n   size_t msg_len = sizeof(BT_HDR) + payload_size + L2CAP_MIN_OFFSET;\n   BT_HDR* p_msg = (BT_HDR*)osi_calloc(msg_len);\n   uint8_t* p = (uint8_t*)(p_msg + 1) + L2CAP_MIN_OFFSET;\n@@ -1049,6 +1065,11 @@ static void gatts_process_read_req(tGATT_TCB& tcb, uint16_t cid,\n                                    uint8_t* p_data) {\n   uint16_t payload_size = gatt_tcb_get_payload_size_tx(tcb, cid);\n \n+  // This can happen if the channel is already closed.\n+  if (payload_size == 0) {\n+    return;\n+  }\n+\n   size_t buf_len = sizeof(BT_HDR) + payload_size + L2CAP_MIN_OFFSET;\n   uint16_t offset = 0;"""

# Output path and filename
OUTPUT_DIR = "saved_inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = "TEST2.diff"
filepath = os.path.join(OUTPUT_DIR, filename)

# Write the string to the file
with open(filepath, "w") as f:
    f.write(input_string)

print(f"✅ Saved input to: {filepath}")
