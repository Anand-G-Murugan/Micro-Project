import numpy as np
import sklearn
import pickle 
import streamlit as st
import os

svr_model=pickle.load(open('models\svr_model.sav','rb')) 
knn_model=pickle.load(open('models\knn_model.sav','rb'))
linear_reg_model=pickle.load(open('models\linear_reg_model.sav','rb'))



def main():

    #url of the website  
    st.title('Salary Prediction')

    #user data 
    # Job Title', 'Rating', 'Company Name', 'Location', 'Size',
    #    'Type of ownership', 'Industry', 'Sector', 'Revenue', 'python_yn',
    #    'R_yn', 'spark', 'aws', 'excel'
    
    job_title=st.selectbox(('Choose your job title'),('Data Scientist',
 'Healthcare Data Scientist',
 'Research Scientist',
 'Staff Data Scientist - Technology',
 'Data Analyst',
 'Data Engineer I',
 'Scientist I/II, Biology',
 'Customer Data Scientist',
 'Data Scientist - Health Data Analytics',
 'Senior Data Scientist / Machine Learning',
 'Data Scientist - Quantitative',
 'Digital Health Data Scientist',
 'Associate Data Analyst',
 'Clinical Data Scientist',
 'Data Scientist / Machine Learning Expert',
 'Web Data Analyst',
 'Senior Data Scientist',
 'Data Engineer',
 'Data Scientist - Algorithms & Inference',
 'Scientist',
 'Lead Data Scientist',
 'Spectral Scientist/Engineer',
 'College Hire - Data Scientist - Open to December 2019 Graduates',
 'Data Scientist, Office of Data Science',
 'Data Science Analyst',
 'Data Scientist (Warehouse Automation)',
 'Jr. Data Scientist',
 'Data Architect / Data Modeler',
 'Associate Scientist / Sr. Associate Scientist, Antibody Discovery',
 'Machine Learning Engineer (NLP)'))
    
    rating=st.number_input('Enter Rating')

    location=st.selectbox(('choose your location'),('Albuquerque, NM',
 'Linthicum, MD',
 'Clearwater, FL',
 'Richland, WA',
 'New York, NY',
 'Dallas, TX',
 'Baltimore, MD',
 'San Jose, CA',
 'Rochester, NY',
 'Chantilly, VA',
 'Plano, TX',
 'Seattle, WA',
 'Cambridge, MA',
 'Newark, NJ',
 'Mountain View, CA',
 'San Francisco, CA',
 'Denver, CO',
 'Chicago, IL',
 'Louisville, KY',
 'Herndon, VA',
 'Hillsboro, OR',
 'Worcester, MA',
 'Groton, CT',
 'Detroit, MI',
 'Sunnyvale, CA',
 'Ipswich, MA',
 'Redlands, CA',
 'Woburn, MA',
 'Fremont, CA',
 'Long Beach, NY',
 'Marlborough, MA',
 'Allendale, NJ',
 'Washington, DC',
 'Bellevue, WA',
 'Longmont, CO',
 'Beavercreek, OH',
 'Peoria, IL',
 'Fort Lauderdale, FL',
 'Boston, MA',
 'Huntsville, AL',
 'Armonk, NY',
 'San Diego, CA',
 'Saint Louis, MO',
 'Cincinnati, OH',
 'Palo Alto, CA',
 'Coraopolis, PA',
 'Framingham, MA',
 'Atlanta, GA',
 'Philadelphia, PA',
 'Vancouver, WA',
 'Indianapolis, IN',
 'Lake Forest, IL',
 'Maryland Heights, MO',
 'Charlottesville, VA',
 'Pittsburgh, PA',
 'Harrisburg, PA',
 'Laurel, MD',
 'Arlington, VA',
 'Tacoma, WA',
 'Miami, FL',
 'New Orleans, LA',
 'Landover, MD',
 'Patuxent River, MD',
 'Suitland, MD',
 'McLean, VA',
 'Fort Belvoir, VA',
 'Milwaukee, WI',
 'Silver Spring, MD',
 'Syracuse, NY',
 'Houston, TX',
 'Charlotte, NC',
 'Southfield, MI',
 'Matawan, NJ',
 'Phoenix, AZ',
 'Omaha, NE',
 'Lyndhurst, NJ',
 'Atlanta, IN',
 'Rockville, MD',
 'Minneapolis, MN',
 'Los Angeles, CA',
 'Alabaster, AL',
 'Santa Fe Springs, Los Angeles, CA',
 'Kansas City, MO',
 'Ashburn, VA',
 'Fort Worth, TX',
 'Valencia, CA',
 'Novato, CA',
 'Aurora, CO',
 'Tampa, FL',
 'Riverton, UT',
 'Chattanooga, TN',
 'Ewing, NJ',
 'South San Francisco, CA',
 'Cupertino, CA',
 'Frederick, MD',
 'West Reading, PA',
 'Madison, WI',
 'Dearborn, MI',
 'Winter Park, FL',
 'San Rafael, CA',
 'Hamilton, NJ',
 'Woodbridge, NJ',
 'Lewes, DE',
 'Springfield, MO',
 'Burbank, CA',
 'Newton, MA',
 'Salt Lake City, UT',
 'Lafayette, LA',
 'Annapolis Junction, MD',
 'Highland, CA',
 'Burleson, TX',
 'Hoopeston, IL',
 'Scotts Valley, CA',
 'Knoxville, TN',
 'Millville, DE',
 'Sheboygan, WI',
 'San Mateo, CA',
 'Dayton, OH',
 'Parlier, CA',
 'Meridian, ID',
 'Cherry Hill, NJ',
 'Nashville, TN',
 'Portland, OR',
 'Port Washington, NY',
 'Austin, TX',
 'Providence, RI',
 'Raleigh, NC',
 'Phila, PA',
 'Oakland, CA',
 'Boise, ID',
 'Oak Ridge, TN',
 'Agoura Hills, CA',
 'Pella, IA',
 'San Ramon, CA',
 'Red Bank, NJ',
 'Columbia, SC',
 'Springfield, MA',
 'San Antonio, TX',
 'Portsmouth, VA',
 'West Palm Beach, FL',
 'Exton, PA',
 'Alexandria, VA',
 'Owensboro, KY',
 'Hartford, CT',
 'Orange, CA',
 'Lenexa, KS',
 'Concord, CA',
 'Vail, CO',
 'Natick, MA',
 'Winston-Salem, NC',
 'Richfield, OH',
 'Hampton, VA',
 'Ithaca, NY',
 'Marietta, GA',
 'Quincy, MA',
 'Green Bay, WI',
 'Durham, NC',
 'Clovis, CA',
 'Chandler, AZ',
 'Orlando, FL',
 'Columbia, MO',
 'Westlake, OH',
 'Des Moines, IA',
 'Cedar Rapids, IA',
 'Fort Lee, NJ',
 'Blue Bell, PA',
 'Springfield, VA',
 'Jersey City, NJ',
 'Emeryville, CA',
 'Santa Barbara, CA',
 'Carle Place, NY',
 'King of Prussia, PA',
 'Santa Clara, CA',
 'Brisbane, CA',
 'Foster City, CA',
 'Holyoke, MA',
 'Waltham, MA',
 'Corvallis, OR',
 'Gaithersburg, MD',
 'Bedford, MA',
 'Aliso Viejo, CA',
 'Dublin, CA',
 'Arvada, CO',
 'Franklin, TN',
 'Plymouth Meeting, PA',
 'Allentown, PA',
 'Logan, UT',
 'Birmingham, AL',
 'Reston, VA',
 'Scottsdale, AZ',
 'Bloomington, IL',
 'Alameda, CA',
 'Roanoke, VA',
 'Glen Burnie, MD',
 'Milpitas, CA',
 'Watertown, MA',
 'Cambridge, MD',
 'Irvine, CA',
 'Ann Arbor, MI',
 'Olympia, WA'))

    
    size=st.selectbox(("enter size of the company"), ('501 to 1000 employees', '10000+ employees',
       '1001 to 5000 employees', '51 to 200 employees',
       '201 to 500 employees', '5001 to 10000 employees',
       '1 to 50 employees', 'Unknown', '-1'))


    ownership=st.selectbox("type of ownership",('Company - Private',
 'Other Organization',
 'Government',
 'Company - Public',
 'Hospital',
 'Subsidiary or Business Segment',
 'Nonprofit Organization',
 'Unknown',
 'College / University',
 'School / School District',
 '-1'))


    industry=st.selectbox(("choose the type of industry"),('Aerospace & Defense',
 'Health Care Services & Hospitals',
 'Security Services',
 'Energy',
 'Advertising & Marketing',
 'Real Estate',
 'Banks & Credit Unions',
 'Consulting',
 'Internet',
 'Other Retail Stores',
 'Research & Development',
 'Department, Clothing, & Shoe Stores',
 'Biotech & Pharmaceuticals',
 'Motion Picture Production & Distribution',
 'Enterprise Software & Network Solutions',
 'Insurance Carriers',
 'Insurance Agencies & Brokerages',
 'Logistics & Supply Chain',
 'Telecommunications Services',
 'IT Services',
 'Computer Hardware & Software',
 '-1',
 'Consumer Products Manufacturing',
 'Industrial Manufacturing',
 'Metals Brokers',
 'Financial Transaction Processing',
 'Sporting Goods Stores',
 'Staffing & Outsourcing',
 'Wholesale',
 'Mining',
 'Financial Analytics & Research',
 'Federal Agencies',
 'Education Training Services',
 'Transportation Equipment Manufacturing',
 'Farm Support Services',
 'TV Broadcast & Cable Networks',
 'Architectural & Engineering Services',
 'Brokerage Services',
 'Travel Agencies',
 'Religious Organizations',
 'Colleges & Universities',
 'Investment Banking & Asset Management',
 'Lending',
 'Gambling',
 'Food & Beverage Manufacturing',
 'Gas Stations',
 'Transportation Management',
 'Video Games',
 'Trucking',
 'Social Assistance',
 'Auctions & Galleries',
 'K-12 Education',
 'Telecommunications Manufacturing',
 'Stock Exchanges',
 'Construction',
 'Accounting',
 'Health Care Products Manufacturing',
 'Health, Beauty, & Fitness',
 'Consumer Product Rental',
 'Beauty & Personal Accessories Stores'))

    
    sector=st.selectbox(('choose sector'),('Aerospace & Defense',
 'Health Care',
 'Business Services',
 'Oil, Gas, Energy & Utilities',
 'Real Estate',
 'Finance',
 'Information Technology',
 'Retail',
 'Biotech & Pharmaceuticals',
 'Media',
 'Insurance',
 'Transportation & Logistics',
 'Telecommunications',
 '-1',
 'Manufacturing',
 'Mining & Metals',
 'Government',
 'Education',
 'Agriculture & Forestry',
 'Travel & Tourism',
 'Non-Profit',
 'Arts, Entertainment & Recreation',
 'Construction, Repair & Maintenance',
 'Accounting & Legal',
 'Consumer Services'))

    
    revenue=st.selectbox(('enter revenue of the company'), ('$50 to $100 million (USD)', '$2 to $5 billion (USD)',
       '$100 to $500 million (USD)', '$500 million to $1 billion (USD)',
       'Unknown / Non-Applicable', '$1 to $2 billion (USD)',
       '$25 to $50 million (USD)', '$10+ billion (USD)',
       '$1 to $5 million (USD)', '$10 to $25 million (USD)',
       '$5 to $10 billion (USD)', 'Less than $1 million (USD)',
       '$5 to $10 million (USD)', '-1'))

    python_yn=st.checkbox('I am proficient in python')
    R_yn=st.checkbox('I am proficient in R')
    aws_yn=st.checkbox('I am proficient in AWS')
    spark_yn=st.checkbox('I am proficient in Spark')
    excel_yn=st.checkbox('I am proficient in Excel')



    if st.button("Submit"):

        with open('encoder_pickles\LabelEncoder_Job Title', 'rb') as file:
            job_title_encoder = pickle.load(file)
        encoded_job_title = job_title_encoder.transform([job_title])
        # st.write(encoded_job_title)

        with open('encoder_pickles\LabelEncoder_Location', 'rb') as file:
            location_encoder = pickle.load(file)
        encoded_location = location_encoder.transform([location])
        # st.write(encoded_location)

        with open('encoder_pickles\LabelEncoder_Size', 'rb') as file:
            size_encoder = pickle.load(file)
        encoded_size = size_encoder.transform([size])
        # st.write(encoded_size)

        with open('encoder_pickles\LabelEncoder_Type_of_ownership', 'rb') as file:
            ownership_encoder = pickle.load(file)
        encoded_ownership = ownership_encoder.transform([ownership])
        # st.write(encoded_ownership)

        with open('encoder_pickles\LabelEncoder_Industry', 'rb') as file:
            industry_encoder = pickle.load(file)
        encoded_industry = industry_encoder.transform([industry])
        # st.write(encoded_industry)

        with open('encoder_pickles\LabelEncoder_Sector', 'rb') as file:
            sector_encoder = pickle.load(file)
        encoded_sector = sector_encoder.transform([sector])
        # st.write(encoded_sector)

        with open('encoder_pickles\LabelEncoder_Revenue', 'rb') as file:
            revenue_encoder = pickle.load(file)
        encoded_revenue = revenue_encoder.transform([revenue])
        # st.write(encoded_revenue)

        input = [encoded_job_title[0], rating, encoded_location[0], encoded_size[0], encoded_ownership[0], encoded_industry[0], 
             encoded_sector[0], encoded_revenue[0], python_yn, R_yn, spark_yn, aws_yn, excel_yn]


        # st.write(input)

        svr_result = svr_model.predict([input])
        knn_result = knn_model.predict([input])
        linear_result = linear_reg_model.predict([input])
        # amount_in_thousands = round(svr_result[0], 2)
        # amount = int(amount_in_thousands * 1000)
        # st.write("You should be making around ${amount:.2f} per year!".format(amount = amount))

        # amount_in_thousands = round(knn_result[0], 2)
        # amount = int(amount_in_thousands * 1000)
        # st.write("You should be making around ${amount:.2f} per year!".format(amount = amount))

        # amount_in_thousands = round(linear_result[0], 2)
        # amount = int(amount_in_thousands * 1000)
        # st.write("You should be making around ${amount:.2f} per year!".format(amount = amount))


        result = (svr_result + knn_result + linear_result) / 3
        amount_in_thousands = round(result[0], 2)
        amount = int(amount_in_thousands * 1000)
        st.write("You should be making around ${amount:.2f} per year!".format(amount = amount))

        # st.write("Submitted!")

if __name__ == '__main__':
    main()
