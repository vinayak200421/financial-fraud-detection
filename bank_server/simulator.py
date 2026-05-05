import requests
import time
import argparse

BASE_URL = "http://127.0.0.1:5000/api"

def login_and_get_token(username, password):
    print(f"Logging in as {username}...")
    resp = requests.post(f"{BASE_URL}/login", json={"username": username, "password": password})
    if resp.status_code == 200:
        print("Login successful!")
        return resp.cookies
    else:
        print("Login failed:", resp.json())
        return None

def trigger_whale(cookies, target):
    print(f"\n--- TRIGGERING WHALE ATTACK (Target: {target}) ---")
    payload = {
        "receiver_username": target,
        "amount": 95000.0,
        "purpose": "Transfer"
    }
    resp = requests.post(f"{BASE_URL}/transfer", json=payload, cookies=cookies)
    print("Response Status:", resp.status_code)
    print("Response Data:", resp.json())

def trigger_velocity(cookies, target, count):
    print(f"\n--- TRIGGERING VELOCITY ATTACK (Target: {target}, Count: {count}) ---")
    for i in range(count):
        payload = {
            "receiver_username": target,
            "amount": 2.50,
            "purpose": f"Micro test {i}"
        }
        print(f"Sending micro-txn {i+1}...")
        resp = requests.post(f"{BASE_URL}/transfer", json=payload, cookies=cookies)
        print("  ->", resp.json())
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank Server Research Simulator")
    parser.add_argument("--strategy", choices=["whale", "velocity", "both"], default="both")
    parser.add_argument("--target_user", default="user_2", help="Username of the receiver")
    parser.add_argument("--burst_count", type=int, default=7, help="Number of transactions for velocity strategy")
    args = parser.parse_args()

    # Login as admin for whale (needs high balance), user_1 for velocity
    whale_cookies = login_and_get_token("admin", "adminpass")
    velocity_cookies = login_and_get_token("user_1", "pass123")
    cookies = velocity_cookies  # default for backward compat
    
    if whale_cookies and args.strategy in ["whale", "both"]:
        trigger_whale(whale_cookies, args.target_user)
    
    if velocity_cookies and args.strategy in ["velocity", "both"]:
        trigger_velocity(velocity_cookies, args.target_user, args.burst_count)
