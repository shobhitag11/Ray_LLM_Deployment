# Ray_LLM_Deployment

1. To deploy llama2-13b-chat model:
serve run -h 0.0.0.0 -p 8001 deploy_llama2_model:deployment

2. To deploy flan-ul2 model:
serve run -h 0.0.0.0 -p 8002 deploy_flan_ul2:deployment

3. To deploy mistral model:
serve run -h 0.0.0.0 -p 8003 deploy_mistral_model:deployment

4. To deploy cross encoder model:
serve run -h 0.0.0.0 -p 8004 deploy_cross_encoder:deployment

# To deploy multi application we can use ray serve multi-app feature
#### Using this feature all the models can be hosted on a single port at different route prefixs. But to do this the machine should be highly capable.
1. serve build --multi-app deploy_llama2_model:deployment deploy_flan_ul2:deployment deploy_mistral_model:deployment deploy_cross_encoder:deployment -o ray_config.yaml
2. serve run ray_config.yaml

# TODO: 
#### Add requirements.txt