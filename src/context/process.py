from typing import List, Optional

from src.models import ConversionResult, SummaryResult
from src.strategies.base import ProcessingStrategy
from src.context.input import ProcessingInput
from src.strategies.factory import StrategyFactory



class DocumentProcessor:
    """Context class for the Strategy Pattern that processes documents.
    
    This class maintains a reference to a strategy object and delegates
    the actual processing work to the strategy. Clients can swap strategies
    at runtime to change how documents are processed.
    
    The processor works with the ProcessingInput model which handles validation
    and strategy type determination. If no strategy is explicitly provided,
    an appropriate one will be automatically selected using the StrategyFactory.
    
    Attributes:
        strategy: The current processing strategy
        primary_docs: List of primary documents to process
        supporting_docs: List of supporting documents to process
        has_medical_records: Flag indicating if medical records are present
        job_id: Unique identifier for the current job
        include_exhibits_research: Flag for including exhibits research
    
    Example:
        ```python
        # Initialize with documents (strategy will be auto-selected)
        processor = DocumentProcessor(
            primary_docs=primary_documents,
            supporting_docs=supporting_documents
        )
        
        # Process documents
        result = processor.process()
        
        # Or explicitly set a strategy
        processor.set_strategy(MedicalOnlyStrategy())
        result = processor.process()
        ```
    """
    
    def __init__(
        self,
        strategy: Optional[ProcessingStrategy] = None,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
        has_medical_records: bool = False,
        job_id: Optional[str] = None,
        include_exhibits_research: bool = False,
        model_info_path: str = "./models.json",
    ):
        """Initialize the DocumentProcessor.
        
        Args:
            strategy: The processing strategy to use
            primary_docs: List of primary documents to process
            supporting_docs: List of supporting documents to process
            has_medical_records: Flag indicating if medical records are present
            job_id: Unique identifier for the current job
            include_exhibits_research: Flag for including exhibits research
            model_info_path: Path to the model information file
        """
        self.strategy = strategy
        self.primary_docs = primary_docs
        self.supporting_docs = supporting_docs
        self.has_medical_records = has_medical_records
        self.job_id = job_id
        self.include_exhibits_research = include_exhibits_research
    
    
    def process(self, input_data: ProcessingInput) -> SummaryResult:
        """Process documents using the current strategy.
        
        This method coordinates document processing by:
        1. Ensuring a valid strategy exists (using the factory if needed)
        2. Delegating processing to the strategy
        3. Returning the results
        
        If no strategy is set, one will be automatically created using
        the StrategyFactory based on the input_data.
        
        Args:
            input_data: Validated ProcessingInput containing all documents
                and processing parameters
                
        Returns:
            SummaryResult: The complete result of document processing
            
        Raises:
            ValueError: If no strategy can be determined
            
        Example:
            ```python
            # Process with existing strategy
            input_data = ProcessingInput(primary_docs=[doc1, doc2])
            result = processor.process(input_data)
            
            # Or create a new ProcessingInput with different parameters
            new_input = ProcessingInput(
                primary_docs=[doc3],
                supporting_docs=[doc4, doc5],
                include_exhibits_research=True
            )
            result = processor.process(new_input)
            ```
        """
        if not self.strategy:        
            # Create appropriate strategy
            self.strategy = StrategyFactory.create_strategy(input_data)
        
        # Process using selected strategy
        return self.strategy.process(input_data)
    

    def set_strategy(self, strategy: ProcessingStrategy):
        """Sets the processing strategy for the DocumentProcessor.
        
        Args:
            strategy: The ProcessingStrategy to set
        """
        self.strategy = strategy