"""
Blockchain and Governance Service
==================================

Implements decentralized governance, token economics, and blockchain logging.
"""

import hashlib
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from core.logger import get_logger
from core.config import get_config

logger = get_logger(__name__)
config = get_config()


@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: float
    data: Dict[str, Any]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine block with proof of work"""
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()


@dataclass
class Transaction:
    """Token transaction"""
    from_address: str
    to_address: str
    amount: float
    timestamp: float
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AIBlockchain:
    """
    Blockchain for AI-Nexus governance and token economics
    """
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_transactions: List[Transaction] = []
        self.mining_reward = config.get('blockchain.tokens.reward_per_task', 10)
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("Blockchain initialized")
    
    def _create_genesis_block(self):
        """Create the first block"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data={'message': 'Genesis Block - AI-Nexus'},
            previous_hash='0'
        )
        genesis.hash = genesis.calculate_hash()
        self.chain.append(genesis)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """Add a new block to the chain"""
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=self.get_latest_block().hash
        )
        
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        
        logger.info(f"Block {new_block.index} added with hash {new_block.hash[:16]}...")
        return new_block
    
    def is_chain_valid(self) -> bool:
        """Validate the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check hash
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Block {i} hash mismatch")
                return False
            
            # Check link
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Block {i} chain broken")
                return False
        
        return True
    
    def add_transaction(self, transaction: Transaction):
        """Add transaction to pending pool"""
        self.pending_transactions.append(transaction)
    
    def mine_pending_transactions(self, mining_reward_address: str):
        """Mine all pending transactions"""
        # Add all pending transactions to block
        block_data = {
            'transactions': [t.to_dict() for t in self.pending_transactions]
        }
        
        block = self.add_block(block_data)
        
        # Reset pending and add reward
        self.pending_transactions = [
            Transaction(
                from_address="system",
                to_address=mining_reward_address,
                amount=self.mining_reward,
                timestamp=time.time()
            )
        ]
        
        return block


class TokenManager:
    """
    Manages AI-Nexus tokens (AINEX)
    """
    
    def __init__(self):
        self.balances: Dict[str, float] = defaultdict(float)
        self.blockchain = AIBlockchain()
        
        # Initial token distribution
        initial_supply = config.get('blockchain.tokens.initial_supply', 1000000000)
        self.balances['system'] = initial_supply
        
        logger.info(f"Token manager initialized with supply: {initial_supply}")
    
    def get_balance(self, address: str) -> float:
        """Get token balance for address"""
        return self.balances[address]
    
    def transfer(self, from_address: str, to_address: str, amount: float) -> bool:
        """Transfer tokens between addresses"""
        if self.balances[from_address] < amount:
            logger.warning(f"Insufficient balance for {from_address}")
            return False
        
        self.balances[from_address] -= amount
        self.balances[to_address] += amount
        
        # Record transaction
        transaction = Transaction(
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            timestamp=time.time()
        )
        self.blockchain.add_transaction(transaction)
        
        logger.info(f"Transferred {amount} tokens from {from_address} to {to_address}")
        return True
    
    def reward_node(self, node_address: str, amount: float):
        """Reward node for completing tasks"""
        self.transfer('system', node_address, amount)
        logger.info(f"Rewarded node {node_address} with {amount} tokens")
    
    def mine_block(self, miner_address: str):
        """Mine pending transactions"""
        block = self.blockchain.mine_pending_transactions(miner_address)
        
        # Update balances based on mined transactions
        for transaction in self.blockchain.pending_transactions:
            self.balances[transaction.to_address] += transaction.amount
        
        return block


class GovernanceSystem:
    """
    Decentralized governance for AI-Nexus
    """
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.proposals: Dict[str, Dict] = {}
        self.votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
        self.voting_period = config.get('blockchain.governance.voting_period', 604800)
        self.quorum = config.get('blockchain.governance.quorum', 0.51)
        self.proposal_threshold = config.get('blockchain.governance.proposal_threshold', 1000)
        
        logger.info("Governance system initialized")
    
    def create_proposal(
        self,
        proposer_address: str,
        title: str,
        description: str,
        proposal_type: str = "general"
    ) -> str:
        """Create a new governance proposal"""
        # Check if proposer has enough tokens
        if self.token_manager.get_balance(proposer_address) < self.proposal_threshold:
            raise ValueError("Insufficient tokens to create proposal")
        
        proposal_id = hashlib.sha256(
            f"{title}{time.time()}".encode()
        ).hexdigest()[:16]
        
        self.proposals[proposal_id] = {
            'id': proposal_id,
            'proposer': proposer_address,
            'title': title,
            'description': description,
            'type': proposal_type,
            'created_at': time.time(),
            'expires_at': time.time() + self.voting_period,
            'status': 'active',
            'yes_votes': 0,
            'no_votes': 0
        }
        
        logger.info(f"Proposal {proposal_id} created by {proposer_address}")
        return proposal_id
    
    def vote(self, proposal_id: str, voter_address: str, vote_yes: bool):
        """Cast a vote on a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError("Proposal not found")
        
        proposal = self.proposals[proposal_id]
        
        if time.time() > proposal['expires_at']:
            raise ValueError("Voting period expired")
        
        # Weight vote by token balance
        voting_power = self.token_manager.get_balance(voter_address)
        
        # Record vote
        self.votes[proposal_id][voter_address] = vote_yes
        
        if vote_yes:
            proposal['yes_votes'] += voting_power
        else:
            proposal['no_votes'] += voting_power
        
        logger.info(f"Vote cast on {proposal_id} by {voter_address}: {vote_yes}")
    
    def tally_votes(self, proposal_id: str) -> Dict[str, Any]:
        """Tally votes and determine proposal outcome"""
        if proposal_id not in self.proposals:
            raise ValueError("Proposal not found")
        
        proposal = self.proposals[proposal_id]
        
        total_votes = proposal['yes_votes'] + proposal['no_votes']
        total_supply = self.token_manager.balances['system']
        
        participation = total_votes / total_supply if total_supply > 0 else 0
        
        # Check quorum
        if participation < self.quorum:
            proposal['status'] = 'failed_quorum'
            result = 'failed'
        elif proposal['yes_votes'] > proposal['no_votes']:
            proposal['status'] = 'passed'
            result = 'passed'
        else:
            proposal['status'] = 'rejected'
            result = 'rejected'
        
        logger.info(f"Proposal {proposal_id} {result} - Participation: {participation:.2%}")
        
        return {
            'proposal_id': proposal_id,
            'result': result,
            'yes_votes': proposal['yes_votes'],
            'no_votes': proposal['no_votes'],
            'participation': participation,
            'status': proposal['status']
        }
