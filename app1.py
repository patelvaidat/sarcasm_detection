import streamlit as st
import warnings
import time
import random
import os
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# Import sarcasm models
from sarcasm_models import (
    predict_sarcasm_xgboost,
    predict_sarcasm_random_forest,
    predict_sarcasm_naive_bayes,
)

# Streamlit UI with fancy styling
st.set_page_config(page_title="Sarcasm Detector", page_icon="🧮", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #1e1e2f;
            color: #ffffff;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50; /* Green color */
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("**NLP-Powered Sarcasm Detector**")
st.subheader("_Let us figure out if you're being sarcastic!_")

# Create two side-by-side columns
col1, col2 = st.columns([2, 3])

# Left column: User input and predictions
with col1:
    st.header("Sarcasm Detection")
    model_choice = st.selectbox(
        "Choose a sarcasm detection model:",
        ("🤖 XGBoost", "🌲 Random Forest", "📊 Naive Bayes")
    )
    user_input = st.text_area("Enter a text to check for sarcasm:", placeholder="Type something sarcastic... or not!")

    if st.button("🔥 Detect Sarcasm 🔥"):
        if user_input:
            if model_choice == "🤖 XGBoost":
                result = predict_sarcasm_xgboost(user_input)
                st.success(f"🎉 **XGBoost Prediction:** {result}")
            
            elif model_choice == "🌲 Random Forest":
                result = predict_sarcasm_random_forest(user_input)
                st.success(f"🌿 **Random Forest Prediction:** {result}")
            
            elif model_choice == "📊 Naive Bayes":
                result = predict_sarcasm_naive_bayes(user_input)
                st.success(f"📉 **Naive Bayes Sarcasm Probability:** {result:.2f}%")
            
            # Random fun messages
            fun_messages = [
                "😜 Keep up the sass!",
                "😂 That was peak sarcasm!",
                "🤨 Was that really sarcastic, though?",
                "🔥 AI is judging your tone!",
                "💡 Sarcasm level: Expert!",
                "🙃 That was dripping with sarcasm!",
                "🎭 Are you auditioning for a sarcasm role?"
            ]
            st.info(random.choice(fun_messages))
        else:
            st.warning("⚠️ Please enter a text.")

# Right column: Display meme dynamically
with col2:
    st.header("Meme of the Moment")
    meme_folder = "memes"

    if os.path.exists(meme_folder) and os.listdir(meme_folder):
        meme_files = sorted([f for f in os.listdir(meme_folder) if f.endswith((".png", ".jpg", ".jpeg", ".gif"))])

        if meme_files:  # Ensure meme_files is not empty
            meme_container = st.empty()  # Create a container for dynamic updates
            meme_index = int(time.time() / 5) % len(meme_files)  # Change meme every 5 seconds
            meme_path = os.path.join(meme_folder, meme_files[meme_index])
            image = Image.open(meme_path).resize((300, 300))
            meme_container.image(image, caption="Random Meme 😂", use_container_width=True)
    else:
        st.warning("⚠️ No memes found in the folder.")

# Meme Gallery at the bottom
st.markdown("---")
st.header("🎨 Meme Gallery")

if os.path.exists(meme_folder) and os.listdir(meme_folder):
    meme_files = sorted([f for f in os.listdir(meme_folder) if f.endswith((".png", ".jpg", ".jpeg", ".gif"))])
    
    cols = st.columns(3)  # Display memes in 3 columns
    for i, meme_file in enumerate(meme_files):
        meme_path = os.path.join(meme_folder, meme_file)
        image = Image.open(meme_path).resize((200, 200))  # Resize for gallery
        with cols[i % 3]:  # Distribute images evenly across columns
            st.image(image, caption=f"Meme {i+1}", use_container_width=True)
else:
    st.warning("⚠️ No memes found for the gallery.")


# Add some unexpected fun at the end
st.markdown("---")
st.header("🤫 Secret Sarcasm Corner")

if st.button("Click for a Secret Sarcasm Dose! 🤯"):
    secret_jokes = [
        "Oh wow, you clicked the button. Such a groundbreaking move. 👏",
        "Sarcasm? Me? Never. I'm as serious as a clown at a funeral. 🤡",
        "Oh, you needed AI to detect sarcasm? That's adorable. 🤗",
        "Breaking news: People are still struggling to detect sarcasm. 😲",
        "Your sarcasm detector is broken. Oh wait, that's just you. 😏",
        "Oh sure, I'm totally paying attention. Keep talking. 🥱",
        "You’re right! Because the internet is *always* correct. 🙃",
    ]
    st.success(random.choice(secret_jokes))

st.markdown("*(Keep this between us, okay? 🤫)*")
