import matplotlib.pyplot as plt
import os


VISUALIZATION_DIR = '/mnt/data/visualizations/'
# Ensure the visualization directory exists
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)


def visualize_power_usage(decision_count, power_usage_history, action_size, reward_history):
    plt.figure(figsize=(10, 6))
    
    # Plot power usage
    plt.subplot(2, 2, 1)
    plt.plot(power_usage_history, label='Power Usage')
    plt.xlabel('Decision Count')
    plt.ylabel('Power Usage')
    plt.title(f'Power Usage (Action Size: {action_size})')
    plt.legend()
    
    # Plot reward history
    plt.subplot(2, 2, 2)
    plt.plot(reward_history, label='Reward per Decision')
    plt.xlabel('Decision Count')
    plt.ylabel('Reward')
    plt.title('Reward per Decision')
    plt.legend()
    
    # # Highlight VMM ID changes
    # plt.subplot(2, 2, 3)
    # vmmid_changes_x, vmmid_changes_y = zip(*vmmid_changes)
    # plt.scatter(vmmid_changes_x, vmmid_changes_y, c='red', label='VMM ID Change', marker='x')
    # plt.xlabel('Decision Count')
    # plt.ylabel('VMM ID')
    # plt.title('VMM ID Changes')
    # plt.legend()
    
    plt.tight_layout()
    
    # Save the figure with a unique name based on the action size and decision count
    filename = f'visualization_action_size_{action_size}_decisions_{decision_count}.png'
    filepath = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Visualization saved to {filepath}")
