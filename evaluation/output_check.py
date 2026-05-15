import pandas as pd
df = pd.read_csv('/workspace/instructblip_outputs/instructblip_results.csv')

print('Per image breakdown:')
for _, row in df.iterrows():
    filename = row['image_path'].split('/')[-1]
    true = row['true_label']
    resnet = row['resnet_pred']
    blip_gender = row['gender_neutral']
    
    if blip_gender == true:
        outcome = 'BLIP CORRECT - avoids ResNet bias'
    elif blip_gender == resnet:
        outcome = 'BLIP WRONG - reproduces ResNet stereotype'
    else:
        outcome = 'BLIP NEUTRAL/AMBIGUOUS'
        
    print(f'{filename}')
    print(f'  True:{true} | ResNet:{resnet} | BLIP:{blip_gender} | {outcome}')
    print()
