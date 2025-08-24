import pandas as pd

def allocate_tasks_with_rebalancing(tasks_df, resources_df, rebalancing_iterations=5):
    """
    Allocates tasks to resources using a greedy approach followed by an iterative rebalancing step.

    Args:
        tasks_df (pd.DataFrame): DataFrame containing task information.
        resources_df (pd.DataFrame): DataFrame containing resource information.
        rebalancing_iterations (int): Number of iterations to run the rebalancing phase.

    Returns:
        pd.DataFrame: A DataFrame with the final, rebalanced allocation results.
    """
    tasks_df = tasks_df.copy()
    resources_df = resources_df.copy()
    
    # Rename columns for clarity
    tasks_df.rename(columns={'nota': 'task_id', 'esforco': 'task_effort', 'prazo': 'prazo'}, inplace=True)
    resources_df.rename(columns={'matricula': 'resource_id', 'nome': 'resource_name', 'disponibilidade': 'disponibilidade'}, inplace=True)
    
    # Calculate slack (folga) for each task and sort
    tasks_df['folga'] = tasks_df['prazo'] - tasks_df['task_effort']
    tasks_df.sort_values(by='folga', ascending=True, inplace=True)
    
    # --- Phase 1: Initial Greedy Allocation ---
    resource_states = {
        row['resource_id']: {
            'name': row['resource_name'],
            'disponibilidade': row['disponibilidade'],
            'current_time': 0,
            'total_effort': 0,
            'proportional_load': 0
        }
        for _, row in resources_df.iterrows()
    }
    allocation_results = []
    
    for _, task in tasks_df.iterrows():
        task_id = task['task_id']
        task_effort_remaining = task['task_effort']
        
        # Find the best resource for the current task
        best_resource_id = min(resource_states, key=lambda res_id: resource_states[res_id]['proportional_load'])
        res_state = resource_states[best_resource_id]
        current_time = res_state['current_time']
        
        # Allocate task, splitting across days if necessary
        while task_effort_remaining > 0:
            day_start_time = (current_time // 1440) * 1440
            time_in_day = current_time % 1440
            daily_capacity_remaining = res_state['disponibilidade'] - time_in_day
            
            effort_to_do = min(task_effort_remaining, daily_capacity_remaining)
            start_time = current_time
            end_time = start_time + effort_to_do
            
            task_effort_remaining -= effort_to_do
            current_time = end_time
            
            if task_effort_remaining > 0 and (end_time % 1440 == res_state['disponibilidade']):
                current_time = day_start_time + 1440
            
            allocation_results.append({
                'task_id': task_id,
                'resource_id': best_resource_id,
                'resource_name': res_state['name'],
                'start_time': start_time,
                'end_time': end_time,
                'task_effort': task['task_effort'],
                'prazo': task['prazo'],
                'folga': task['folga']
            })
        
        res_state['total_effort'] += task['task_effort']
        res_state['current_time'] = current_time
        res_state['proportional_load'] = res_state['total_effort'] / res_state['disponibilidade']

    # Convert initial results to DataFrame for rebalancing
    initial_allocation_df = pd.DataFrame(allocation_results)

    # --- Phase 2: Iterative Rebalancing ---
    rebalanced_allocation_df = initial_allocation_df.copy()
    
    for _ in range(rebalancing_iterations):
        for task_id in tasks_df['task_id'].unique():
            # Get current task allocation details
            current_task_rows = rebalanced_allocation_df[rebalanced_allocation_df['task_id'] == task_id]
            if current_task_rows.empty:
                continue
            
            current_resource_id = current_task_rows['resource_id'].iloc[0]
            current_task_effort = current_task_rows['task_effort'].iloc[0]
            
            # Find a potential new resource
            best_new_resource_id = None
            min_proportional_load = resource_states[current_resource_id]['proportional_load']
            
            for res_id, res_state in resource_states.items():
                if res_id != current_resource_id:
                    # Calculate new proportional load if task is moved
                    new_resource_load_after = res_state['total_effort'] + current_task_effort
                    new_proportional_load = new_resource_load_after / res_state['disponibilidade']
                    
                    if new_proportional_load < min_proportional_load:
                        # Check if moving the task would violate folga order for the new resource
                        is_valid_move = True
                        current_task_folga = tasks_df[tasks_df['task_id'] == task_id]['folga'].iloc[0]
                        tasks_on_new_resource = rebalanced_allocation_df[rebalanced_allocation_df['resource_id'] == res_id]
                        
                        for _, existing_task_row in tasks_on_new_resource.iterrows():
                            existing_task_folga = tasks_df[tasks_df['task_id'] == existing_task_row['task_id']]['folga'].iloc[0]
                            # A task with higher folga cannot be started before a task with lower folga.
                            # So, if we move a task to a resource, its folga must be handled correctly.
                            # A simple rule: if a task with higher folga is already assigned, we can't move
                            # a task with lower folga to it if it would violate the order.
                            # The simplest way to handle this in rebalancing is to allow the move only if the
                            # new task's folga is lower than the folga of all tasks already on the new resource.
                            if current_task_folga > existing_task_folga:
                                is_valid_move = False
                                break
                        
                        if is_valid_move:
                            best_new_resource_id = res_id
                            min_proportional_load = new_proportional_load

            # If a better resource is found, reassign the task
            if best_new_resource_id:
                # Update resource states
                resource_states[current_resource_id]['total_effort'] -= current_task_effort
                resource_states[current_resource_id]['proportional_load'] = resource_states[current_resource_id]['total_effort'] / resource_states[current_resource_id]['disponibilidade']
                
                resource_states[best_new_resource_id]['total_effort'] += current_task_effort
                resource_states[best_new_resource_id]['proportional_load'] = resource_states[best_new_resource_id]['total_effort'] / resource_states[best_new_resource_id]['disponibilidade']
                
                # Update the allocation DataFrame
                rebalanced_allocation_df.loc[rebalanced_allocation_df['task_id'] == task_id, 'resource_id'] = best_new_resource_id
                rebalanced_allocation_df.loc[rebalanced_allocation_df['task_id'] == task_id, 'resource_name'] = resource_states[best_new_resource_id]['name']

    # Recalculate start and end times after rebalancing
    final_allocation_results = []
    
    # Reset resource times for the final pass
    for res_id in resource_states:
        resource_states[res_id]['current_time'] = 0

    # Group tasks by new resource and sort by folga for accurate scheduling
    for resource_id in rebalanced_allocation_df['resource_id'].unique():
        resource_tasks = rebalanced_allocation_df[rebalanced_allocation_df['resource_id'] == resource_id].drop_duplicates(subset='task_id').sort_values(by='folga')
        
        for _, task in resource_tasks.iterrows():
            task_id = task['task_id']
            task_effort_remaining = task['task_effort']
            res_state = resource_states[resource_id]
            current_time = res_state['current_time']
            
            while task_effort_remaining > 0:
                day_start_time = (current_time // 1440) * 1440
                time_in_day = current_time % 1440
                daily_capacity_remaining = res_state['disponibilidade'] - time_in_day
                
                effort_to_do = min(task_effort_remaining, daily_capacity_remaining)
                start_time = current_time
                end_time = start_time + effort_to_do
                
                task_effort_remaining -= effort_to_do
                current_time = end_time
                
                if task_effort_remaining > 0 and (end_time % 1440 == res_state['disponibilidade']):
                    current_time = day_start_time + 1440
                
                final_allocation_results.append({
                    'task_id': task_id,
                    'resource_id': resource_id,
                    'resource_name': res_state['name'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'task_effort': task['task_effort'],
                    'prazo': task['prazo'],
                    'folga': task['folga']
                })
            
            res_state['current_time'] = current_time

    final_df = pd.DataFrame(final_allocation_results)
    final_df['esforco_total'] = final_df['resource_id'].map({res_id: res_state['total_effort'] for res_id, res_state in resource_states.items()})
    final_df['antecedencia'] = final_df['prazo'] - final_df['end_time']

    return final_df

# --- Main script ---
if __name__ == '__main__':
    # Load data from the specified paths
    try:
        resources_df = pd.read_csv('.\\data\\prd_rot_recursos.csv', sep=';')
        tasks_df = pd.read_csv('.\\data\\prd_rot_tarefas.csv', sep=';')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the files are in the specified path.")
        raise

    # Run the allocation algorithm with rebalancing
    final_allocation_df = allocate_tasks_with_rebalancing(tasks_df, resources_df, rebalancing_iterations=10)

    # Save the final DataFrame to a CSV file
    output_file_name = '.\\data\\saida_greedy_prioridade_balanceamento.csv'
    final_allocation_df.to_csv(output_file_name, index=False, sep=';')

    print(f"Allocation complete. Results saved to '{output_file_name}'.")

    # Display the first few rows of the final output
    print("\nFinal Allocation DataFrame (with rebalancing):")
    print(final_allocation_df.head())