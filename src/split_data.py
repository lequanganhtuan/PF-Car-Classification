import splitfolders

input_folder = "./data/train_organized" 

output_folder = "./data/final_data"

# Thực hiện split
splitfolders.ratio(
    input_folder, 
    output = output_folder, 
    seed = 42,           
    ratio = (.8, .2),    
    group_prefix = None, 
    move = False         
)

print("--- Finish ---")