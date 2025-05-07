import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import time
import random

# Load the pre-trained model (would need to be saved from the notebook)
# model = load_model('parking_classifier.h5')

# Mock functions since we don't have the actual model/data
def count_parking_slots():
    """Mock function to count available/occupied slots"""
    total_slots = 100
    occupied = random.randint(40, 70)
    available = total_slots - occupied
    return total_slots, available, occupied

def predict_parking_status(image_path):
    """Mock function to predict if a parking slot is occupied"""
    # In a real app, this would use the actual model
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # prediction = model.predict(img_array)
    # return "Occupied" if prediction[0][0] > 0.5 else "Available"
    return random.choice(["Available", "Occupied"])

def generate_receipt(reservation_data):
    """Generate a receipt for the reservation"""
    receipt = f"""
    PARKING RECEIPT
    ----------------------------
    Reservation ID: {reservation_data['reservation_id']}
    Date: {reservation_data['date']}
    Time: {reservation_data['time']}
    ----------------------------
    Customer Name: {reservation_data['name']}
    Vehicle Number: {reservation_data['vehicle_number']}
    Slot Number: {reservation_data['slot_number']}
    Duration: {reservation_data['duration']} hours
    ----------------------------
    Amount Paid: ${reservation_data['amount']}
    Payment Method: {reservation_data['payment_method']}
    ----------------------------
    Thank you for using our service!
    """
    return receipt

def main():
    st.title("🏢 Smart Parking Management System")
    
    # Sidebar navigation
    menu = ["Parking Status", "Reserve a Slot", "Payment", "View Receipt"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Parking status page
    if choice == "Parking Status":
        st.header("Current Parking Status")
        
        # Get parking counts (would use model in real implementation)
        total, available, occupied = count_parking_slots()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Slots", total)
        with col2:
            st.metric("Available", available, delta=f"{-occupied} occupied")
        with col3:
            st.metric("Occupied", occupied, delta=f"{available} available", delta_color="inverse")
        
        # Display a mock parking lot visualization
        st.subheader("Parking Lot Overview")
        fig, ax = plt.subplots(figsize=(10, 5))
        status = ['Available', 'Occupied']
        counts = [available, occupied]
        ax.bar(status, counts, color=['green', 'red'])
        ax.set_title('Parking Slot Status')
        ax.set_ylabel('Number of Slots')
        st.pyplot(fig)
        
        # Upload image for single spot detection (mock)
        st.subheader("Check Specific Parking Spot")
        uploaded_file = st.file_uploader("Upload an image of a parking spot", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display the image
            st.image(uploaded_file, caption="Uploaded Parking Spot", width=300)
            
            # Predict status
            status = predict_parking_status("temp_image.jpg")
            st.write(f"Predicted Status: **{status}**")
            
            # Remove temp file
            os.remove("temp_image.jpg")
    
    # Reservation page
    elif choice == "Reserve a Slot":
        st.header("Reserve a Parking Slot")
        
        # Get current parking status
        total, available, _ = count_parking_slots()
        
        if available == 0:
            st.warning("Sorry, no parking slots are currently available.")
        else:
            with st.form("reservation_form"):
                st.write("Please fill in the reservation details:")
                
                name = st.text_input("Full Name")
                vehicle_number = st.text_input("Vehicle Registration Number")
                duration = st.selectbox("Duration (hours)", [1, 2, 3, 4, 5, 6, 8, 12, 24])
                date = st.date_input("Date of Reservation", min_value=datetime.today())
                time = st.time_input("Time of Arrival")
                
                # Generate a random slot number (in real app, would check availability)
                slot_number = random.randint(1, total)
                
                submitted = st.form_submit_button("Check Availability")
                
                if submitted:
                    if not name or not vehicle_number:
                        st.error("Please fill in all required fields")
                    else:
                        # Calculate cost ($2 per hour)
                        amount = duration * 2
                        
                        # Store reservation data in session state
                        st.session_state.reservation_data = {
                            "name": name,
                            "vehicle_number": vehicle_number,
                            "duration": duration,
                            "date": date.strftime("%Y-%m-%d"),
                            "time": str(time),
                            "slot_number": slot_number,
                            "amount": amount,
                            "reservation_id": f"RES-{random.randint(1000, 9999)}"
                        }
                        
                        st.success(f"Slot #{slot_number} is available for your selected time!")
                        st.write(f"Estimated cost: ${amount}")
                        st.info("Proceed to payment to confirm your reservation")
    
    # Payment page
    elif choice == "Payment":
        st.header("Payment Information")
        
        if 'reservation_data' not in st.session_state:
            st.warning("Please make a reservation first")
        else:
            reservation = st.session_state.reservation_data
            
            st.write("Reservation Summary:")
            st.write(f"- Slot Number: {reservation['slot_number']}")
            st.write(f"- Date: {reservation['date']}")
            st.write(f"- Time: {reservation['time']}")
            st.write(f"- Duration: {reservation['duration']} hours")
            st.write(f"- Total Amount: ${reservation['amount']}")
            
            with st.form("payment_form"):
                st.write("Payment Details:")
                
                payment_method = st.selectbox("Payment Method", 
                                            ["Credit Card", "Debit Card", "PayPal", "Mobile Payment"])
                card_number = st.text_input("Card Number", disabled=payment_method in ["PayPal", "Mobile Payment"])
                expiry = st.text_input("Expiry Date (MM/YY)", disabled=payment_method in ["PayPal", "Mobile Payment"])
                cvv = st.text_input("CVV", disabled=payment_method in ["PayPal", "Mobile Payment"], type="password")
                
                pay_now = st.form_submit_button("Pay Now")
                
                if pay_now:
                    if payment_method in ["Credit Card", "Debit Card"] and (not card_number or not expiry or not cvv):
                        st.error("Please enter all card details")
                    else:
                        # Mock payment processing
                        with st.spinner("Processing payment..."):
                            time.sleep(2)
                        
                        # Add payment method to reservation data
                        reservation['payment_method'] = payment_method
                        st.session_state.reservation_data = reservation
                        
                        # Generate receipt
                        st.session_state.receipt = generate_receipt(reservation)
                        
                        st.success("Payment successful! Your reservation is confirmed.")
                        st.info("Go to 'View Receipt' to see and download your receipt")
    
    # Receipt page
    elif choice == "View Receipt":
        st.header("Your Parking Receipt")
        
        if 'receipt' not in st.session_state:
            st.warning("No receipt available. Please complete a reservation and payment.")
        else:
            st.code(st.session_state.receipt)
            
            # Download button for receipt
            st.download_button(
                label="Download Receipt",
                data=st.session_state.receipt,
                file_name=f"parking_receipt_{st.session_state.reservation_data['reservation_id']}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
