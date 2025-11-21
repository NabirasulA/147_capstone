import numpy as np
import pandas as pd
import tensorflow as tf

class VANETDataGenerator(tf.keras.utils.Sequence):
    """A data generator class for VANET data that inherits from tf.keras.utils.Sequence.
    This class is designed to handle large datasets efficiently by loading data in batches.
    """
    
    def __init__(self, data_path, batch_size=32, sequence_length=10, test_split=0.3, is_training=True):
        """Initialize the data generator.
        
        Args:
            data_path (str): Path to the CSV file containing VANET data
            batch_size (int): Number of samples per batch
            sequence_length (int): Length of the sequence for each sample
            test_split (float): Proportion of data to use for validation (between 0 and 1)
            is_training (bool): Whether this generator is for training or validation data
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.test_split = test_split
        self.is_training = is_training
        
        # Get total number of rows
        self.total_rows = sum(1 for _ in open(data_path)) - 1  # Subtract header row
        
        # Calculate split indices
        self.split_idx = int(self.total_rows * (1 - test_split))
        
        # Set start and end indices based on training/validation split
        if is_training:
            self.start_idx = 0
            self.end_idx = self.split_idx
        else:
            self.start_idx = self.split_idx
            self.end_idx = self.total_rows
            
        self.num_samples = self.end_idx - self.start_idx
        
        # Calculate number of batches
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.num_batches += 1
            
        # Initialize chunk reading state
        self.current_chunk = None
        self.current_chunk_start = None
        self.current_chunk_end = None
        self.chunk_size = batch_size * 100  # Read 100 batches worth of data at a time
        
        # Get feature names (assuming first row is header)
        self.feature_names = pd.read_csv(data_path, nrows=0).columns.tolist()

        # Detect label column with common names, fall back to last column
        possible_labels = ['label', 'Label', 'attack', 'ATTACK', 'attack_type', 'target', 'class']
        detected = None
        for cand in possible_labels:
            if cand in self.feature_names:
                detected = cand
                break
        if detected is None:
            detected = self.feature_names[-1]

        self.label_column = detected
        # Exclude common non-numeric columns from features
        exclude_cols = set([
            self.label_column,
            'type', 'rcvTime', 'rcv_time', 'timestamp', 'id', 'index', ''
        ])
        self.feature_columns = [col for col in self.feature_names if col not in exclude_cols]
        
    def __len__(self):
        """Return the number of batches per epoch."""
        return self.num_batches
    
    def __getitem__(self, idx):
        """Get batch at position idx.
        
        Args:
            idx (int): Position of the batch in the sequence.
            
        Returns:
            tuple: (input features, labels) for the batch
        """
        # Calculate batch boundaries
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        
        # Check if we need to load new chunk
        if (self.current_chunk is None or 
            start < self.current_chunk_start or 
            end > self.current_chunk_end):
            
            # Calculate chunk boundaries
            self.current_chunk_start = start
            self.current_chunk_end = min(start + self.chunk_size, self.end_idx)
            
            # Read chunk of data. Handle header correctly to avoid reading header row as data.
            read_nrows = self.current_chunk_end - self.current_chunk_start
            if self.current_chunk_start == 0:
                # Let pandas read header from the file normally
                self.current_chunk = pd.read_csv(
                    self.data_path,
                    nrows=read_nrows
                )
            else:
                # Skip the header row and previous rows, read without header and assign names
                self.current_chunk = pd.read_csv(
                    self.data_path,
                    skiprows=range(1, self.current_chunk_start + 1),
                    nrows=read_nrows,
                    header=None
                )
                self.current_chunk.columns = self.feature_names
        
        # Extract batch data from current chunk
        batch_start = start - self.current_chunk_start
        batch_end = end - self.current_chunk_start
        batch_data = self.current_chunk.iloc[batch_start:batch_end]
        
        # Prepare sequences
        sequences = []
        labels = []

        # Coerce feature columns to numeric, non-convertible values become NaN then filled with 0.0
        features_df = batch_data[self.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        for i in range(len(batch_data) - self.sequence_length + 1):
            sequence = features_df.iloc[i:i + self.sequence_length].values
            label = batch_data[self.label_column].iloc[i + self.sequence_length - 1]
            sequences.append(sequence)
            labels.append(label)
            
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(labels)

        # Ensure shape is (batch, seq_len, num_features)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=2)

        # Cast to numeric types for TF
        try:
            X = X.astype(np.float32)
        except Exception:
            # If conversion fails, attempt to coerce via pandas
            X = np.array(batch_data[self.feature_columns].astype(float).values)
            if X.ndim == 2:
                X = np.expand_dims(X, axis=2)
            X = X.astype(np.float32)

        try:
            y = y.astype(np.int32)
        except Exception:
            # Map labels if they are strings (e.g., 'attack'/'benign')
            unique_labels, y_encoded = np.unique(y, return_inverse=True)
            y = y_encoded.astype(np.int32)

        return X, y
    
    def on_epoch_end(self):
        """Called at the end of every epoch."""
        pass  # We don't need to do anything here for this implementation