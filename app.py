import streamlit as st
import matplotlib.pyplot as plt
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def predict_text_classification_single_label_sample(
    project: str,
    endpoint_id: str,
    content: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = predict.instance.TextClassificationPredictionInstance(
        content=content,
    ).to_value()
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # print("response")
    # print(" deployed_model_id:", response.deployed_model_id)

    predictions = response.predictions
    for prediction in predictions:
        l = dict(prediction)
        return l['displayNames'] , l['confidences']
    
def split_result(result):
    a = list(result[0])
    b = list(result[1])
    return a,b

def bar_chart(result):
    
    a =result[0]
    b =result[1]

    a= list(a)
    b= list(b)

    fig = plt.figure(figsize = (10, 5))

    plt.bar(a, b)
    plt.xlabel("labels")
    plt.ylabel("confidence")
    plt.title("News Detector")
    st.pyplot(fig)
    
def main():
    st.title("Fake News Detection ")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">FAKE NEWS DETECTOR </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    test_input = st.text_area("INPUT","type here")
    result = predict_text_classification_single_label_sample(
        project="820931443216",
        endpoint_id="2269435980194775040",
        location="us-central1",
        content=test_input)
   
    
    if st.button("Predict"):
       
        
        a = list(result[0])
        b = list(result[1])
        pos = b.index(max(list(b)))
        
        #st.text("The above content happens to be: ", a[pos])
     
        bar_chart(result)
    
    
       
    
    #st.success("The above news is:",a[pos])
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with love with Streamlit")
    
if __name__ == "__main__":
    main()