import gradio as gr
import boto3
import json
import threading
import webbrowser

# CONFIGURATIE
ENDPOINT_NAME = "mushroom-endpoint"
REGION = "eu-west-1"               

# AWS Client setup
sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)

modelName = "AWS-SageMaker-RF"

def query_endpoint(payload):
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read().decode())
        return result
    except Exception as e:
        return {"error": str(e)}

def collect_data(cap_diameter, stem_height, stem_width, gill_spacing, 
                 does_bruise_bleed, has_ring, cap_shape, 
                 cap_surface, stem_surface, cap_color, gill_color, stem_color, veil_color, spore_print_color,
                 gill_attachment, stem_root, ring_type, 
                 habitat, season):
    payload = {
        "cap_diameter": cap_diameter,
        "stem_height": stem_height,
        "stem_width": stem_width,
        "gill_spacing": gill_spacing,
        "does_bruise_bleed": does_bruise_bleed,
        "has_ring": has_ring,
        "cap_shape": cap_shape,
        "cap_surface": cap_surface,
        "stem_surface": stem_surface,
        "cap_color": cap_color,
        "gill_color": gill_color,
        "stem_color": stem_color,
        "veil_color": veil_color,
        "spore_print_color": spore_print_color,
        "gill_attachment": gill_attachment,
        "stem_root": stem_root,
        "ring_type": ring_type if has_ring else 4, # Logica voor ring type
        "habitat": habitat,
        "season": season
    }
    response_data = query_endpoint(payload)

    if "error" in response_data:
        return f"System Error: {response_data['error']}"

    # krijg  het antwoord van de container
    # formaat: {"prediction": "edible", "confidence": 42.5}
    pred = response_data.get("prediction", "Unknown")
    conf = response_data.get("confidence", 0.0)
    return f"This mushroom is : {pred}, Confidence: {conf}%"

# --- GRADIO INTERFACE ---
with gr.Blocks() as app:
    gr.Markdown("# Mushroom Data Interface ( laatste test !!!)")
    
    # SLIDERS
    cap_diameter = gr.Slider(0.38, 62.34, value=10.0, label="Cap diameter (cm)")
    stem_height = gr.Slider(0.0, 33.92, value=5.0, label="Stem height (cm)")
    stem_width = gr.Slider(0.0, 103.91, value=10.0, label="Stem width (mm)")
    
    # RADIO BUTTONS
    gill_spacing = gr.Radio([("close", 0), ("distant", 1)], label="Gill spacing", value=0)
    
    # CHECKBOXES
    does_bruise_bleed = gr.Checkbox(label="Does it bruise or bleed ?")
    has_ring = gr.Checkbox(label="Has ring")
    
    # DROPDOWNS (Waardes zijn integers, labels zijn tekst)
    cap_shape = gr.Dropdown([("bell", 3), ("conical", 4), ("convex", 0), ("flat", 1), ("sunken", 5), ("spherical", 2), ("others", 6)], label="Cap shape", value=0)
    cap_surface = gr.Dropdown([("dry", 7), ("fibrous", 9), ("grooves", 0), ("scaly", 3), ("smooth", 5), ("shiny", 1), ("leathery", 6), ("silky", 10), ("sticky", 2), ("wrinkled", 8), ("fleshy", 4)], label="Cap Surface", value=7)
    stem_surface = gr.Dropdown([("dry", 7), ("fibrous", 9), ("grooves", 0), ("scaly", 3), ("smooth", 5), ("shiny", 1), ("leathery", 6), ("silky", 10), ("sticky", 2), ("wrinkled", 8), ("fleshy", 4)], label="Stem Surface", value=7)
    
    # Colors
    color_options = [("brown", 2), ("buff", 9), ("gray", 3), ("green", 4), ("pink", 7), ("purple", 8), ("red", 1), ("white", 5), ("yellow", 6), ("blue", 10), ("orange", 0), ("black", 11)]
    cap_color = gr.Dropdown(color_options, label="Cap color", value=2)
    gill_color = gr.Dropdown(color_options, label="Gill color", value=2)
    stem_color = gr.Dropdown(color_options, label="Stem color", value=2)
    veil_color = gr.Dropdown(color_options, label="Veil color", value=2)
    spore_print_color = gr.Dropdown(color_options, label="Spore print color", value=2)
    
    gill_attachment = gr.Dropdown([("adnate", 1), ("adnexed", 4), ("decurrent", 2), ("free", 0), ("sinuate", 3), ("pores", 5), ("No_attachment", 6)], label="Gill attachment", value=1)
    stem_root = gr.Dropdown([("bulbous", 1), ("swollen", 0), ("club", 3), ("Filamentous", 4), ("rooted", 2)], label="Stem root", value=1)
    
    ring_type = gr.Dropdown([("evanescent", 2), ("flaring", 6), ("grooved", 0), ("large", 3), ("pendant", 1), ("zone", 7), ("movable", 5)], label="Ring type", visible=False, value=2)
    
    habitat = gr.Dropdown([("grasses", 2), ("leaves", 4), ("meadows", 1), ("paths", 5), ("heaths", 3), ("urban", 7), ("waste", 6), ("woods", 0)], label="Habitat", value=0)
    season = gr.Dropdown([("spring", 3), ("summer", 1), ("autumn", 2), ("winter", 0)], label="Season", value=3)
    
    # Output
    output = gr.Textbox(label="Result from AWS SageMaker")
    
    # Submit button
    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=collect_data,
        inputs=[cap_diameter, stem_height, stem_width, gill_spacing, 
              does_bruise_bleed, has_ring, cap_shape, 
              cap_surface, stem_surface, cap_color, gill_color, stem_color, veil_color, spore_print_color,
              gill_attachment, stem_root, ring_type, 
              habitat, season],
        outputs=output
    )                        

    # Dynamisch veld (Ring Type)
    has_ring.change(fn=lambda x: gr.update(visible=x), inputs=has_ring, outputs=ring_type)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)