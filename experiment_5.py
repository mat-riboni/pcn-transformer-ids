# %% [markdown]
# # Recurrent PCN

# %%
from utils.data_utils import *
from utils.preprocessing_utils import *
from PCN import *

categorical_features = [
    "L4_SRC_PORT",           
    "L4_DST_PORT",
    "PROTOCOL",              
    "L7_PROTO",
    "TCP_FLAGS",             
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "ICMP_TYPE",             
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",          
    "DNS_QUERY_TYPE",        
    "FTP_COMMAND_RET_CODE"   
]

numerical_features = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",                   
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",            
    "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER"             
    ]

T_infer = 100

X, y = load_dataset("archive/NF-UNSW-NB15-v2.csv")

# %%
removed = False
if not removed:
    X = remove_ip_fields(X)
    removed = True #we can re-run the cell
X_train, X_test, y_train, y_test = split_dataset_temporal(X, y, test_size=0.2)
#X_train, y_ssl = create_ssl_dataset(X_train, y_train, label_ratio=0.9999)

print(X_train.shape)
#print(y_ssl.head())
#print(y_ssl.value_counts())

# %%
X_train = cap_numerical_data(X_train, numerical_features)
X_test = cap_numerical_data(X_test, numerical_features)

X_train, X_test, min_max_scaler = min_max_log_norm(X_train, X_test, numerical_features)
X_train, X_test, categories_dict = keep_top_categorical_level(X_train, X_test, categorical_features, max_levels=32)
print(categories_dict)

X_train, X_test, one_hot_encoder = ordinal_encode_categorical(X_train, X_test, categorical_features)
print(X_train.head())

# %% [markdown]
# # Predictive Coding Network

# %%
from torch import float32
from torch.utils.data import DataLoader
from PCN.trainer import train_pcn_binary


from PCN.PCNetwork import PredictiveCodingNetwork

device = 'cuda'
pcn = PredictiveCodingNetwork([41, 64, 32])
X_tensor = torch.tensor(np.array(X_train), dtype=float32).to(device)
print("X tensor ok")
y_tensor = torch.tensor(np.array(y_train), dtype=float32).to(device)
print("y tensor ok")
print(X_train.shape)
print(y_train.shape)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=4096, shuffle=True)
print("trainloader")
train_pcn_binary(model=pcn, data_loader=train_loader, num_epochs=40, eta_infer=0.01, eta_learn=0.0005, T_infer=T_infer, margin_attack=500, device=device)
torch.save(pcn.state_dict(), 'pcn_model_weights_2.pth')

# %%
from utils.train_utils import  evaluate_pcn_anomaly
from PCN.PCNetwork import PredictiveCodingNetwork
device = 'cuda'

pcn_loaded = PredictiveCodingNetwork([41, 64, 32])
state_dict = torch.load('pcn_model_weights_2.pth', map_location=device)
pcn_loaded.load_state_dict(state_dict)
pcn_loaded.to(device)
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32).view(-1, 1).to(device)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=2048, shuffle=False)
pcn_loaded.eval()

evaluate_pcn_anomaly(pcn_loaded, test_loader, T_infer=T_infer, eta_infer=0.01, threshold_energy=5, device=device)


