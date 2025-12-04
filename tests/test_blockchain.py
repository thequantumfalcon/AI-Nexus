"""
Test suite for Blockchain Service
==================================

Tests blockchain, token economics, and governance.
"""

import pytest
import time

from services.blockchain.blockchain import (
    Block,
    Transaction,
    AIBlockchain,
    TokenManager,
    GovernanceSystem
)


class TestBlock:
    """Test blockchain block"""
    
    def test_block_creation(self):
        """Test creating a block"""
        block = Block(
            index=1,
            timestamp=time.time(),
            data={'message': 'test'},
            previous_hash='0' * 64
        )
        
        assert block.index == 1
        assert block.data == {'message': 'test'}
    
    def test_block_hash_calculation(self):
        """Test block hash calculation"""
        block = Block(
            index=1,
            timestamp=time.time(),
            data={'test': 'data'},
            previous_hash='0' * 64
        )
        
        hash1 = block.calculate_hash()
        hash2 = block.calculate_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
    
    def test_block_mining(self):
        """Test proof-of-work mining"""
        block = Block(
            index=1,
            timestamp=time.time(),
            data={'test': 'data'},
            previous_hash='0' * 64
        )
        
        difficulty = 2
        block.mine_block(difficulty)
        
        assert block.hash.startswith('0' * difficulty)
        assert block.nonce > 0


class TestAIBlockchain:
    """Test blockchain functionality"""
    
    @pytest.fixture
    def blockchain(self):
        """Create blockchain fixture"""
        return AIBlockchain(difficulty=2)
    
    def test_genesis_block(self, blockchain):
        """Test genesis block creation"""
        assert len(blockchain.chain) == 1
        assert blockchain.chain[0].index == 0
        assert blockchain.chain[0].previous_hash == '0'
    
    def test_add_block(self, blockchain):
        """Test adding blocks"""
        initial_length = len(blockchain.chain)
        
        blockchain.add_block({'transaction': 'test'})
        
        assert len(blockchain.chain) == initial_length + 1
        assert blockchain.chain[-1].data == {'transaction': 'test'}
    
    def test_chain_linking(self, blockchain):
        """Test blocks are properly linked"""
        blockchain.add_block({'data': 'block1'})
        blockchain.add_block({'data': 'block2'})
        
        block1 = blockchain.chain[1]
        block2 = blockchain.chain[2]
        
        assert block2.previous_hash == block1.hash
    
    def test_chain_validation(self, blockchain):
        """Test blockchain validation"""
        blockchain.add_block({'data': 'test1'})
        blockchain.add_block({'data': 'test2'})
        
        assert blockchain.is_chain_valid()
    
    def test_chain_tampering_detection(self, blockchain):
        """Test detecting tampered blocks"""
        blockchain.add_block({'data': 'test'})
        
        # Tamper with data
        blockchain.chain[1].data = {'data': 'tampered'}
        
        assert not blockchain.is_chain_valid()
    
    def test_transaction_mining(self, blockchain):
        """Test mining transactions"""
        tx = Transaction(
            from_address="alice",
            to_address="bob",
            amount=100,
            timestamp=time.time()
        )
        
        blockchain.add_transaction(tx)
        block = blockchain.mine_pending_transactions("miner")
        
        assert len(blockchain.chain) > 1


class TestTokenManager:
    """Test token management"""
    
    @pytest.fixture
    def token_manager(self):
        """Create token manager fixture"""
        return TokenManager()
    
    def test_initialization(self, token_manager):
        """Test token manager initialization"""
        assert token_manager.balances['system'] > 0
        assert len(token_manager.blockchain.chain) == 1
    
    def test_get_balance(self, token_manager):
        """Test getting balance"""
        balance = token_manager.get_balance('system')
        assert balance > 0
    
    def test_transfer(self, token_manager):
        """Test token transfer"""
        # Give alice some tokens first
        token_manager.balances['alice'] = 1000
        
        success = token_manager.transfer('alice', 'bob', 100)
        
        assert success
        assert token_manager.get_balance('alice') == 900
        assert token_manager.get_balance('bob') == 100
    
    def test_insufficient_balance(self, token_manager):
        """Test transfer with insufficient balance"""
        token_manager.balances['charlie'] = 50
        
        success = token_manager.transfer('charlie', 'dave', 100)
        
        assert not success
        assert token_manager.get_balance('charlie') == 50
    
    def test_reward_node(self, token_manager):
        """Test node reward"""
        initial_balance = token_manager.get_balance('node1')
        
        token_manager.reward_node('node1', 50)
        
        assert token_manager.get_balance('node1') == initial_balance + 50


class TestGovernanceSystem:
    """Test governance functionality"""
    
    @pytest.fixture
    def governance(self):
        """Create governance system fixture"""
        token_manager = TokenManager()
        return GovernanceSystem(token_manager)
    
    def test_create_proposal(self, governance):
        """Test creating proposal"""
        # Give proposer enough tokens
        governance.token_manager.balances['proposer'] = 2000
        
        proposal_id = governance.create_proposal(
            'proposer',
            'Test Proposal',
            'This is a test proposal',
            'general'
        )
        
        assert proposal_id in governance.proposals
        assert governance.proposals[proposal_id]['title'] == 'Test Proposal'
    
    def test_proposal_threshold(self, governance):
        """Test proposal creation threshold"""
        governance.token_manager.balances['poor_proposer'] = 100
        
        with pytest.raises(ValueError):
            governance.create_proposal(
                'poor_proposer',
                'Test',
                'Should fail'
            )
    
    def test_voting(self, governance):
        """Test casting votes"""
        # Setup
        governance.token_manager.balances['proposer'] = 2000
        governance.token_manager.balances['voter1'] = 1000
        governance.token_manager.balances['voter2'] = 500
        
        proposal_id = governance.create_proposal(
            'proposer',
            'Test Vote',
            'Description'
        )
        
        # Vote
        governance.vote(proposal_id, 'voter1', True)
        governance.vote(proposal_id, 'voter2', False)
        
        proposal = governance.proposals[proposal_id]
        assert proposal['yes_votes'] == 1000
        assert proposal['no_votes'] == 500
    
    def test_tally_votes_passed(self, governance):
        """Test vote tallying - passed"""
        # Setup with high token supply
        governance.token_manager.balances['system'] = 10000
        governance.token_manager.balances['proposer'] = 2000
        governance.token_manager.balances['voter1'] = 6000
        
        proposal_id = governance.create_proposal(
            'proposer',
            'Pass This',
            'Should pass'
        )
        
        # Vote yes with majority
        governance.vote(proposal_id, 'voter1', True)
        
        result = governance.tally_votes(proposal_id)
        
        assert result['result'] == 'passed'
        assert result['yes_votes'] > result['no_votes']
    
    def test_tally_votes_quorum_failure(self, governance):
        """Test vote tallying - quorum not met"""
        governance.token_manager.balances['system'] = 1000000  # Large supply
        governance.token_manager.balances['proposer'] = 2000
        governance.token_manager.balances['voter1'] = 100
        
        proposal_id = governance.create_proposal(
            'proposer',
            'No Quorum',
            'Should fail quorum'
        )
        
        governance.vote(proposal_id, 'voter1', True)
        
        result = governance.tally_votes(proposal_id)
        
        assert result['result'] == 'failed'
        assert result['participation'] < governance.quorum


@pytest.mark.integration
class TestBlockchainIntegration:
    """Integration tests for blockchain system"""
    
    def test_complete_governance_workflow(self):
        """Test full governance workflow"""
        # Initialize
        token_manager = TokenManager()
        governance = GovernanceSystem(token_manager)
        
        # Distribute tokens
        token_manager.balances['proposer'] = 5000
        token_manager.balances['voter1'] = 10000
        token_manager.balances['voter2'] = 8000
        token_manager.balances['voter3'] = 6000
        
        # Create proposal
        proposal_id = governance.create_proposal(
            'proposer',
            'Network Upgrade',
            'Upgrade to version 2.0'
        )
        
        # Voting
        governance.vote(proposal_id, 'voter1', True)
        governance.vote(proposal_id, 'voter2', True)
        governance.vote(proposal_id, 'voter3', False)
        
        # Tally
        result = governance.tally_votes(proposal_id)
        
        assert result['result'] in ['passed', 'rejected', 'failed']
        assert result['yes_votes'] + result['no_votes'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
