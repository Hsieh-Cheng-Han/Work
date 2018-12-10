#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream> 
#include <ctime>
#include <time.h>
#include <stdio.h>  
#include <typeinfo>
#include <math.h>
#include <algorithm>
////////////////////// Self-Made cpp ////////////////////
#include "myfunction.cpp"
#include "interpolation.cpp"
#include "DM_curve_tran_v2.cpp"
#include "DM_expand_forward_curve.cpp"
#include "CF_portfolio_pricing_info_SP.cpp"
#include "CF_cashflow_iteration.cpp"
#include "CF_Discount_Price_SP.cpp"
#include "AM_cfdates.cpp"
#include "AM_AIfactor.cpp"
#include "AM_CaliamortIRR.cpp"
#include "AM_Amortization.cpp"


using namespace std;

int main(){
	/*Model Input*/

    const int Pricing_Y = 2018;
    const int Pricing_M = 3;
    const int Pricing_D = 31;
    
    const int Curve_Y = 2018;
    const int Curve_M = 3;
    const int Curve_D = 31;   
    const int Frequency = 12;
    const string Curve_interpre_method = "linear";
    const int Projected_year_Frequency = 12;
    const string Protofolio_Data = "Bond_20180331.csv";
    const int Max_tenor = Frequency*60 + 1;
    const int Pricing_method = 1;
    const int Shock_Type = 4;
    const double FX_USD[2] = {31,32};
	 
	double* Date_Data = new double[2];
	Date_Data[0] = datenum(Curve_Y, Curve_M, Curve_D);
	// need to modify
	Date_Data[1] = datenum(2018,4,30);
	
	cout << "Discount Module begins." << endl;
	/*Interest Curve Setup*/
	const char* Curve_Data_Name = "Curve20180330.txt";
	txtfile Curve_Data = loadtxt(Curve_Data_Name);
	vector<string> currency_category = Curve_Data.file[0]; //{t, AUD, CNH...AUDB} size = 25
	const int currency_number = Curve_Data.column_number - 1; // 24
	double** nomin_idx = new2Darray(3,Curve_Data.column_number);
	nomin_idx[0][0] = 1;
	nomin_idx[1][0] = 1;
	nomin_idx[2][0] = 1;
    for(int i = 1;i < Curve_Data.column_number;i++){
		string target = currency_category[i];
		if(target == "EUR"||target == "TWD"||target == "EURT"||target == "TWDT"||target == "CNYT"){
			nomin_idx[0][i] = 1; 
			nomin_idx[1][i] = 0;
	    }
		else{
		    nomin_idx[0][i] = 0;
		    if(target == "CNH"||target == "CNY")
			    nomin_idx[1][i] = 0;
			else
			    nomin_idx[1][i] = 1; 
		}
		if(target == "CNH"||target == "CNY"||target == "AUDB")
			nomin_idx[2][i] = 1; 
		else 
		    nomin_idx[2][i] = 0;		    
	}
    
	/* Sensitivity Curve Setup */
	const char* Sensitivity_Table_Name = "SensitivityTable.txt"; 
	txtfile Sensitivity_Table = loadtxt(Sensitivity_Table_Name);
	
	vector<string> Curve_Shock_txt = Sensitivity_Table.file[1];//{t,AUD,...} size = 25
	
	// Check if currency_category is equal to Curve_Shock_txt
	for(int i = 0; i < currency_number; i++){
		if(currency_category[i] != Curve_Shock_txt[i]){
		    cout << "The Sensitivity Table is not consistent with the Interest Form." << endl;
		    system("pause");
		}  
	} 
	
    double Level_Shock_idx[currency_number];//size = 24
    for(int i = 0; i < currency_number; i++)
	    Level_Shock_idx[i] = (stod(Sensitivity_Table.file[0][i+1]) != 0 ? 0 : 1);	
	
	const int idx_number = Sensitivity_Table.row_number - 2;// 122
    int* idx = new int[idx_number]();//{1,4,7...721} size = 122
	for(int i = 0; i < idx_number;i++){
		idx[i] = int(stod(Sensitivity_Table.file[i+2][0])*12 + 1);
	}

	// Construct a 2D array
	double** Shock_Scen = new2Darray(Max_tenor, currency_number);// 721*24

	for(int i = 0; i < idx_number;i++){
		for(int j = 0; j < currency_number;j++)
            Shock_Scen[idx[i]-1][j] = stod(Sensitivity_Table.file[i+2][j+1])/10000;
	}
	//Set Level Shock Scene
	for(int j = 0; j < currency_number;j++){
		if(Level_Shock_idx[j]){
		    for(int i = 0; i < Max_tenor;i++)
                Shock_Scen[i][j] = stod(Sensitivity_Table.file[0][j+1])/10000;
	    }
	}
	
	cout << "Construct Bootstrapping Curve." << endl;
	const int Curve_value_row_number = Curve_Data.row_number-1;
	const int Curve_value_column_number = Curve_Data.column_number;

	// Construct Curve by Curve_value(first column is the time)
	double** Curve_value = new2Darray(Curve_value_row_number, Curve_value_column_number);
        
	for(int i = 0; i < Curve_value_row_number;i++){
		Curve_value[i][0] = stod(Curve_Data.file[i+1][0]);
		for(int j = 1; j < Curve_value_column_number;j++){
			if(Curve_Data.file[i+1][j] != "NA")
                Curve_value[i][j] = stod(Curve_Data.file[i+1][j])/100;
            else    
                Curve_value[i][j] = 999;// indicate NA
            //cout << Curve_value[i][j] << endl;    
	    }
	}

    // Interpolating missing data
	Curve_value = interpolation(Curve_value, Curve_value_row_number, Curve_value_column_number, "linear");   

	// Survey the number of "true" in nomin_idx
    int nomin_stat[3] = {0,0,0};
	for(int i = 0; i <= currency_number;i++){
		if(nomin_idx[0][i])
		    nomin_stat[0]++;
		if(nomin_idx[1][i])
		    nomin_stat[1]++;
		if(nomin_idx[2][i])
		    nomin_stat[2]++;
	}	
    
    		
	// Initialize eff_spt 2D array(size: 721*25)
	double** eff_spt = new2Darray(Max_tenor, currency_number + 1);
	int expand_length = Frequency / Projected_year_Frequency + 1;
	
	// Choose Shock_Type
	switch(Shock_Type){
		case 4 : {	
		    
			using_DM(eff_spt, Curve_value, nomin_idx, 0, "spot", 1, 1, Max_tenor, 12, "linear");
			using_DM(eff_spt, Curve_value, nomin_idx, 1, "spot", 1, 2, Max_tenor, 12, "linear");
			using_DM(eff_spt, Curve_value, nomin_idx, 2, "spot", 1, 4, Max_tenor, 12, "linear");
           		
			// Put the shock on the effective rate table 
			for(int j = 1; j < Curve_Data.column_number;j++){				
		    	for(int i = 0; i < Max_tenor;i++){
		    		eff_spt[i][j] = eff_spt[i][j] + Shock_Scen[i][j-1];				    
			    }
			}	
			//
			vector<double**> eff_spt_vector; 
			double* eff_spt_1 = new double[Max_tenor]();
			for(int i = 0;i < Max_tenor;i++)
				eff_spt_1[i] = eff_spt[i][0];
			for(int i = 1;i < Curve_Data.column_number;i++){
				double* eff_spt_x = new double[Max_tenor]();
			    for(int j = 0;j < Max_tenor;j++)
				    eff_spt_x[j] = eff_spt[j][i];
				eff_spt_vector.push_back(DM_expand_forward_curve(Frequency,Projected_year_Frequency,eff_spt_x,eff_spt_1));    			
			}			
			//
			eff_spt = new2Darray(Max_tenor,currency_number*expand_length);
			for(int j = 0;j < currency_number*expand_length;j++){
				for(int i = 0;i < Max_tenor;i++)
				    eff_spt[i][j] = eff_spt_vector[j/expand_length][i][j%expand_length];
			}			
			break; 		
        }
        // end case 1
    }
    // end Switch

    double** spot_output = copy2Darray(eff_spt);
    double** forward_output = new2Darray(Max_tenor,currency_number*expand_length + 1);
    double** tmp_data = new2Darray(Max_tenor,currency_number*expand_length + 1);

    for(int i = 0;i < Max_tenor;i++)
		tmp_data[i][0] = double(i)/double(Frequency);	
	for(int j = 1;j < currency_number*expand_length + 1;j++){
		for(int i = 0;i < Max_tenor;i++)
			tmp_data[i][j] = eff_spt[i][j-1];
	}
	
	double** tmp_nomin_idx = copy2Darray(nomin_idx); 
	//slightly different with nomin_idx
	for(int i = 1;i < Curve_Data.column_number;i++){
		string target = currency_category[i];
		if(target == "AUDB")
			tmp_nomin_idx[2][i] = 0; 		    
	}
	nomin_idx = new2Darray(3,currency_number*expand_length + 1);
	for(int j = 0;j < 3;j++)
		nomin_idx[j][0] = 1;
    for(int j = 1;j < currency_number*expand_length + 1;j++){
		for(int i = 0;i < 3;i++)
			nomin_idx[i][j] = tmp_nomin_idx[i][(j-1)/expand_length+1];
	}
	delete tmp_nomin_idx;

	using_DM(forward_output, tmp_data, nomin_idx, 0, "forward", 2, 1, Max_tenor, 12, "linear");
	using_DM(forward_output, tmp_data, nomin_idx, 1, "forward", 2, 2, Max_tenor, 12, "linear");
	using_DM(forward_output, tmp_data, nomin_idx, 2, "forward", 2, 4, Max_tenor, 12, "linear");
	
	delete idx;
	int reshape_idx[3] = {Max_tenor, Frequency/Projected_year_Frequency + 1, currency_number};
	
	double** tmp = new2Darray(Max_tenor, currency_number*expand_length + 1);
	tmp = split2Darray(forward_output, 0, Max_tenor, 1, currency_number*expand_length + 1);

	double*** forward_output_reshape = reshape(tmp, reshape_idx);
	double*** spot_output_reshape = reshape(spot_output, reshape_idx);
	
	double**** forward_output_final = new4Darray(Max_tenor, Frequency/Projected_year_Frequency + 1, 18, currency_number);
	double**** spot_output_final = new4Darray(Max_tenor, Frequency/Projected_year_Frequency + 1, 18, currency_number);
	for(int i = 0; i < Max_tenor;i++){
		    for(int j = 0; j < Frequency/Projected_year_Frequency + 1;j++){
			    for(int k = 0; k < 18;k++){
			    	for(int s = 0; s < currency_number;s++){
				    forward_output_final[i][j][k][s] = forward_output_reshape[i][j][s];
					spot_output_final[i][j][k][s] = spot_output_reshape[i][j][s];				
			    }
			}
		}
	}

	for(int j = 0;j < Max_tenor;j++)
	    tmp[j][0] = double(j)/Frequency;
	for(int j = 0;j < Max_tenor;j++){
		for(int i = 1;i < currency_number*expand_length + 1;i++)
			tmp[j][i] = eff_spt[j][i-1];
	}	
	double** future_forswp = DM_curve_tran_v2(tmp, 2, 4, Max_tenor, 12, "linear").output_par_curve;
		
	tmp = split2Darray(future_forswp, 0, Max_tenor, 1, currency_number*expand_length + 1);
	
	double*** future_forswp_reshape = reshape(tmp, reshape_idx);

	///////////////////////////////////////////////////////////////////////////
	// Volatility Setup //
	const char* Volatility_Table_Name= "CapVolatility.txt";
	txtfile Volatility_Table = loadtxt(Volatility_Table_Name);
	
	const int Vol_row_number = Volatility_Table.row_number - 1;
	const int Vol_column_number = Volatility_Table.column_number;
	
	double** VolData = new2Darray(Vol_row_number, Vol_column_number);
      
	for(int i = 0; i < Vol_row_number;i++){
		VolData[i][0] = stod(Volatility_Table.file[i+1][0]);
		for(int j = 1; j < Vol_column_number;j++){
			if(Volatility_Table.file[i+1][j] != "NA")
                VolData[i][j] = stod(Volatility_Table.file[i+1][j])/100;
            else    
                VolData[i][j] = 999;// indicate NA   
	    }
	}

	string* VolCur = new string[Vol_column_number - 1]();
	for(int i = 0; i < Vol_column_number - 1;i++)
		VolCur[i] = Volatility_Table.file[0][i+1];
	
	int output_tenor_number = 4*(Max_tenor - 1)/Frequency;	
	
	double** VolTermStrct = new2Darray(output_tenor_number, currency_number);
	int* VolCur_Num_tmp = new int[Vol_column_number - 1]();
	for(int i = 0; i < Vol_column_number - 1;i++){
		for(int j = 0; j < currency_number;j++){
	        if(VolCur[i] == currency_category[j+1])
	            VolCur_Num_tmp[i] = j + 1;
	    }
    }
	
	// set up the dataframe for interpolation
	tmp = new2Darray(output_tenor_number, Vol_column_number);
	for(int i = 0; i < output_tenor_number;i++)
		tmp[i][0] = double(i+1)/4;
	for(int j = 1; j < Vol_column_number;j++){
	    for(int i = 0; i < output_tenor_number;i++)
		    tmp[i][j] = searchdata(tmp[i][0], 0, j, VolData); 		
	}	
	tmp = interpolation(tmp, output_tenor_number, Vol_column_number, "spline");	
	for(int j = 0; j < Vol_column_number - 1;j++){
	    for(int i = 0; i < output_tenor_number;i++)
		    VolTermStrct[i][VolCur_Num_tmp[j] - 1] = tmp[i][j+1]; 		
	}	
	delete VolData, VolCur, VolCur_Num_tmp; 
	/////////////////////////////////////////////////////////
	// Protfolio Setup //
	const char* Portfolio_Table_Name= "Bond_20180331.txt";
	txtfile Portfolio_Table = loadtxt(Portfolio_Table_Name);
	
	// basic information set up
	const int start_row = 2056;
	const int end_row = 2066;
	const int port_row_number = end_row - start_row;
	int period = Frequency/Projected_year_Frequency + 1 ;
	double** port_data = new2Darray(port_row_number, 22 + period);
	
	int column_index = indexing(Portfolio_Table.file[0], "MaturityYear");
	for(int i = start_row; i < end_row; i++){	   
	    double tmp = stod(Portfolio_Table.file[i][column_index]);
	    if(tmp > Pricing_Y + 60)
		    Portfolio_Table.file[i][column_index] = to_string(Pricing_Y + 60); 
	}
		
	// 1. Bond type
	column_index = indexing(Portfolio_Table.file[0], "BondType");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][0] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}	
    // 2. Currency
	column_index = indexing(Portfolio_Table.file[0], "Currency");
	for(int i = 0; i < port_row_number; i++){	    
	    for(int j = 0; j < currency_number; j++){
	    	if(currency_category[j] == Portfolio_Table.file[i + start_row][column_index])
		        port_data[i][1] = j;
		}
	}
	// 3. Volatility index
	for(int i = 0; i < port_row_number; i++){	    
	    for(int j = 0; j < currency_number; j++){
	    	if(currency_category[j] == Portfolio_Table.file[i + start_row][column_index].substr(0,3))
		        port_data[i][2] = j;
		}
	}
	// 4. Credit Rating index
	column_index = indexing(Portfolio_Table.file[0], "Rating");
	const string credit_rating_index[18] = {"Treasuries","AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-",
	"BB+","BB","BB-","B+","B","B-","C"}; 
	for(int i = 0; i < port_row_number; i++){	    
	    for(int j = 0; j < 18; j++){
	    	if(credit_rating_index[j] == Portfolio_Table.file[i + start_row][column_index])
		        port_data[i][3] = double(j + 1);
		}
	}
	// 5. Call Flag
	column_index = indexing(Portfolio_Table.file[0], "CallFlag");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][4] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 6. Coupon Frequency
	column_index = indexing(Portfolio_Table.file[0], "CouponFrequency");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][5] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 7. Call Frequency
	column_index = indexing(Portfolio_Table.file[0], "CallFreq");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][6] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 8. Coupon
	column_index = indexing(Portfolio_Table.file[0], "Coupon");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][7] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 9. Initial Price
	column_index = indexing(Portfolio_Table.file[0], "InitialPrice");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][8] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 10. Floating Reference Rate Index
	for(int i = 0; i < port_row_number; i++){
	    port_data[i][9] = port_data[i][1];  	
	    if(port_data[i][0] != 2)    
		    port_data[i][9] = 0;
	}
	// 11. Nominal Frequency
	column_index = indexing(Portfolio_Table.file[0], "Currency");
	for(int i = 0; i < port_row_number; i++){	 
	    string target = Portfolio_Table.file[i + start_row][column_index];   
		if(target == "EUR"||target == "TWD"||target == "EURT"||target == "TWDT"||target == "CNYT")
		    port_data[i][10] = 1;
		if(target != "EUR"&&target != "TWD"&&target != "EURT"&&target != "TWDT"&&target != "CNYT"&&target != "CNH"&&target != "CNY")
		    port_data[i][10] = 2; 
		if(target == "CNH"||target == "CNY")
		    port_data[i][10] = 4;   
	}
	// 12. Floating Bond Spread
	column_index = indexing(Portfolio_Table.file[0], "AddSpread");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][11] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 13. Preset Floating Rate
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][12] = 0.016426;
	}
	// 14. Redemption Price
	column_index = indexing(Portfolio_Table.file[0], "Redemption");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][13] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 15. Last call Price
	column_index = indexing(Portfolio_Table.file[0], "LastCallPrice");
	for(int i = 0; i < port_row_number; i++){	    
		port_data[i][14] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 16. Calculate the end time of the cash flow
	
	//int CountType[port_row_number]; 
	//column_index = indexing(Portfolio_Table.file[0], "CountConvention");
    //for(int i = 0; i < port_row_number; i++){
		//string target = Portfolio_Table.file[i + start_row][column_index]; 
		//if(target == "30/360"||target == "30S/360"||target == "30U/360")    
		    //CountType[i] = 5;
		//if(target == "ACT/360")    
		    //CountType[i] = 9; 
		//if(target == "ACT/365")    
		    //CountType[i] = 10;
		//if(target == "AFB(ACT/ACT)"||target == "ISMA-99(ACT/ACT)")    
		    //CountType[i] = 8;		   
	//}
	tm Pricing_time = {0};
	Pricing_time.tm_year = Pricing_Y - 1900;
	Pricing_time.tm_mon = Pricing_M;
	Pricing_time.tm_mday = Pricing_D;

	
	int column_index_my = indexing(Portfolio_Table.file[0], "MaturityYear");
	int column_index_mm = indexing(Portfolio_Table.file[0], "MaturityMonth");
	int column_index_md = indexing(Portfolio_Table.file[0], "MaturityDay");
	for(int i = 0; i < port_row_number; i++){
	    tm Maturity_time = {0};	
	    Maturity_time.tm_year = stoi(Portfolio_Table.file[i + start_row][column_index_my]) - 1900;    
		Maturity_time.tm_mon = stoi(Portfolio_Table.file[i + start_row][column_index_mm]);
		Maturity_time.tm_mday = stoi(Portfolio_Table.file[i + start_row][column_index_md]);
		// Difference between Pricing time and Maturity time (in year)
		double tmp = difftime(mktime(&Maturity_time), mktime(&Pricing_time))/(365*86400);
		port_data[i][18] = tmp;
		tmp = ceil(tmp*Frequency);
		if(tmp > Max_tenor - 1)
		    tmp = Max_tenor - 1;
		if(tmp == 0)
		    tmp = 1;  
		port_data[i][15] = tmp;  
	}
	// 17. Calculate the begin time of cash flow
	for(int i = 0; i < port_row_number; i++){
	    int tmp = int(port_data[i][15]) % int(port_data[i][5]);
		if(tmp == 0)
		    tmp = int(port_data[i][5]);	    
		port_data[i][16] = double(tmp);
	}
	// 18. Interest Methodology(BDT Tree = 1; IFMR = 2)
	for(int i = 0; i < port_row_number; i++){
	    double tmp = 0;
		if(port_data[i][4] != 0) 
		    tmp = 1;
		if(port_data[i][4] != 0&&port_data[i][0] == 3) 
		    tmp = 2;		    
		port_data[i][17] = tmp;
	}
    // 19. Pricing Information
    switch(Frequency){
    	case 12 : {
    	    for(int i = 0; i < port_row_number; i++){	
    	        port_data[i][18] = stod(Portfolio_Table.file[i + start_row][column_index_md]) / 30;
	            int tmp_month = stoi(Portfolio_Table.file[i + start_row][column_index_mm]);
	            int tmp_year = stoi(Portfolio_Table.file[i + start_row][column_index_my]);
	            if(stoi(Portfolio_Table.file[i + start_row][column_index_my]) > Pricing_Y + 60) 
		            tmp_year = Pricing_Y + 60;  
				if(tmp_month == 1||tmp_month == 3||tmp_month == 5||tmp_month == 7||tmp_month == 8||tmp_month == 10||tmp_month == 12)
				    port_data[i][18] = stod(Portfolio_Table.file[i + start_row][column_index_md]) / 31;
				if(tmp_month == 2)// leap year
    		        port_data[i][18] = stod(Portfolio_Table.file[i + start_row][column_index_md]) / 28; 			
    		    if(tmp_year % 4 == 0&&tmp_month == 2)// leap year
    		        port_data[i][18] = stod(Portfolio_Table.file[i + start_row][column_index_md]) / 29;    

    	    }
			break;
		}
    	default :{
    		for(int i = 0; i < port_row_number; i++)
    		    port_data[i][18] = (port_data[i][18]*Frequency - floor(port_data[i][18]*Frequency))/Frequency;
			break;
		}	
	}
	// 20. Calculate accural interest
	for(int i = 0; i < port_row_number; i++){
	    double numerator = port_data[i][5]*Frequency/12 - (port_data[i][16] - (1-port_data[i][18]));
	    double denominator = port_data[i][5]*Frequency/12;
    	port_data[i][19] = ((numerator/denominator) - int(numerator/denominator))*denominator*port_data[i][7]/Frequency;
    }
    // 21. Call Begin time
    int column_index_cy = indexing(Portfolio_Table.file[0], "CallYear");
	int column_index_cm = indexing(Portfolio_Table.file[0], "CallMonth");
	int column_index_cd = indexing(Portfolio_Table.file[0], "CallDay");
	
	for(int i = 0; i < port_row_number; i++){
	    tm Call_time = {0};	
	    Call_time.tm_year = stoi(Portfolio_Table.file[i + start_row][column_index_cy]) - 1900;      
		Call_time.tm_mon = stoi(Portfolio_Table.file[i + start_row][column_index_cm]);
		Call_time.tm_mday = stoi(Portfolio_Table.file[i + start_row][column_index_cd]);
		// Difference between Pricing time and Call time (in year)
		double tmp = difftime(mktime(&Call_time), mktime(&Pricing_time))/(365*86400);
		tmp = ceil(tmp*Frequency);
		if(tmp > Max_tenor - 1)
		    tmp = Max_tenor - 1;
		if(tmp == 0)
		    tmp = 1;  
		port_data[i][20] = tmp;  
	}
    
    cout << "Portfolio Input Ready !!" << endl;
    //////////////////////////////////////////////////////////////////////
    // Cash Flow module //
    
    // Index for which calculation is needed
    double* cal_idx = new double[port_row_number];
    int cal_number = 0;
    for(int i = 0; i < port_row_number;i++){
    	if(port_data[i][3] != 0 && port_data[i][1] != 0 && port_data[i][2] != 0 && port_data[i][0] != 0){
		    cal_idx[i] = 1;
		    cal_number++;
		}
	}
	
	double** CF = new2Darray(Max_tenor, port_row_number);
    double** Call_Schedule = new2Darray(Max_tenor, port_row_number);
    double*** Forward_Curve = new3Darray(Max_tenor, Frequency/Projected_year_Frequency + 1, port_row_number);
    
    // port_data(cal_idx,:)
    double** port_data_2 = new2Darray(cal_number, 21);	    
    int count = 0;
	for(int i = 0; i < port_row_number; i++){
	    if(cal_idx[i] == 1){
	        for(int j = 0; j < 21; j++)
		        port_data_2[count][j] = port_data[i][j];
            count++;
	    }
	}
	
	// CF(1:max(port_data.CF_endtime),cal_idx)
    vector<double> CF_endtime;
    for(int i = 0; i < port_row_number;i++)
    	CF_endtime.push_back(port_data[i][15]);
    double the_max = *max_element(CF_endtime.begin(), CF_endtime.end());
			   
    switch(Pricing_method){
        case 1 : {
        	
        	CF_INFO cf_result = CF_portfolio_pricing_info_SP(port_data_2, Frequency, future_forswp_reshape, forward_output_final);
        	
        	//put the output of the function into variable
            count = 0;
        	for(int i = 0; i < port_row_number; i++){
	        	if(cal_idx[i] == 1){
	        	    for(int j = 0; j < the_max; j++){
		                CF[j][count] = cf_result.CF[j][i];
		                Call_Schedule[j][count] = cf_result.Call_Schedule[j][i];
		            }
		            for(int j = 0; j < Max_tenor; j++){
		            	for(int k = 0; k < Frequency/Projected_year_Frequency + 1; k++)
		                    Forward_Curve[j][k][count] = cf_result.Forward_Curve[j][k][i];
		            }
                    count++;
	            }
	        }
			break;
		}	
    	case 2 : {
    		// Not Done Already
    		//CF_BDT cf_result = CF_portfolio_pricing_info_bdt(port_data_2, Frequency, spot_output_final, future_forswp_reshape, forward_output_final, VolTermStrct);
    		//count = 0;
        	//for(int i = 0; i < port_row_number; i++){
	        	//if(cal_idx[i] == 1){
	        	    //for(int j = 0; j < the_max; j++){
		                //CF[j][count] = cf_result.CF[j][i];
		                //Call_Schedule[j][count] = cf_result.Call_Schedule[j][i];
		            //}
		            //for(int j = 0; j < Max_tenor; j++){
		            	//for(int k = 0; k < Frequency/Projected_year_Frequency + 1; k++)
		                    //Forward_Curve[j][k][count] = cf_result.Forward_Curve[j][k][i];
		            //}
                    //count++;
	            //}
	        //}
			break;
		}   	
	}
	
	cout << "Pricing Information Ready !!" << endl;
	
    double* im_spread = new double[cal_number]();
	double** Clean_Price = new2Darray(port_row_number, Frequency/Projected_year_Frequency + 1);
	double** Dirty_Price = new2Darray(port_row_number, Frequency/Projected_year_Frequency + 1);
	int* bond_index = new int[cal_number]();
	for(int i = 0; i < cal_number;i++)
	    bond_index[i] = i + 1;  
	int Frac_Year = Frequency/Projected_year_Frequency;    
	double** CF_end_time_pro = new2Darray(port_row_number, Frac_Year + 1);
	double** CF_begin_time_pro = new2Darray(port_row_number, Frac_Year + 1);
	double** Projected_Month = new2Darray(port_row_number, Frac_Year + 1);
	double** Projected_Year = new2Darray(port_row_number, Frac_Year + 1);
	for(int i = 0; i < port_row_number;i++){
		for(int j = 0; j < Frac_Year + 1;j++){
			CF_end_time_pro[i][j] = max(double(0), port_data[i][15] - double(Frequency*j) / double(Projected_year_Frequency));
			CF_begin_time_pro[i][j] = int(CF_end_time_pro[i][j]) % int(port_data[i][5]);
			Projected_Month[i][j] = (stoi(Portfolio_Table.file[i + start_row][column_index_mm]) + j*Frequency/Projected_year_Frequency) % Frequency;
		    if(Projected_Month[i][j] == 0)
		        Projected_Month[i][j] = 12;
		    Projected_Year[i][j] = stoi(Portfolio_Table.file[i + start_row][column_index_my]) + floor(Projected_Month[i][j]/Frequency);    
		}   
	}	
    
    double** cfac_pro = new2Darray(port_row_number, Frac_Year + 1);
    double** call_f_pro = new2Darray(port_row_number, Frac_Year + 1);
    double** spread_pro = new2Darray(cal_number, Frac_Year + 1);
    double** coupon_freq = new2Darray(port_row_number, Frac_Year + 1);
    double** call_freq = new2Darray(port_row_number, Frac_Year + 1);
    double** comf_pro = new2Darray(port_row_number, Frac_Year + 1);
    double** AI_pro = new2Darray(port_row_number, Frac_Year + 1);
    for(int i = 0; i < port_row_number;i++){
		for(int j = 0; j < Frac_Year + 1;j++){
			cfac_pro[i][j] = port_data[i][18]; 
			call_f_pro[i][j] = port_data[i][4];
			coupon_freq[i][j] = port_data[i][5];
			call_freq[i][j] = port_data[i][6];
			comf_pro[i][j] = port_data[i][10]; 
		}
	}
	for(int i = 0; i < cal_number;i++){
		for(int j = 0; j < Frac_Year + 1;j++){
			spread_pro[i][j] = im_spread[i];
		}
	}
	for(int i = 0; i < cal_number;i++){
		for(int j = 0; j < Frac_Year + 1;j++){
			double tmp = port_data[i][5]*Frequency/12 - (CF_begin_time_pro[i][j]-1+cfac_pro[i][j]);
			tmp = tmp - (int(tmp)/int(port_data[i][5]*Frequency/12))*port_data[i][5]*Frequency/12;
			AI_pro[i][j] = tmp * port_data[i][7]/Frequency;
		}
	}
	
	// Cash Flow Iteration for projection(Fixed Bonds & Zero Coupon Bonds)
	double* FloBonds = new double[port_row_number]();
	int Flo_count = 0;
	for(int i = 0; i < port_row_number;i++){
	    if(port_data[i][0] == 2 && cal_idx[i] == 1){
	        FloBonds[i] == 1;	
	        Flo_count++;
	    }
	}
	////////////////////////// Delete it after used //////////////////
	Flo_count = 3;
	FloBonds[0] = 1;
	FloBonds[1] = 1;
	FloBonds[2] = 1;
	/////////////////////////////////////////////////////////////////////
	CF_ITERED cf_itered_result = CF_cashflow_iteration(CF, Call_Schedule, Frac_Year, 1);
    double*** CF_pro = cf_itered_result.CF_itered;
    double*** callT_pro = cf_itered_result.Call_Titered;

			 
	int* flobonds_cur = new int[Flo_count];
	count = 0;
	for(int i = 0; i < port_row_number;i++){
	    if(FloBonds[i] == 1){
	        flobonds_cur[count] = int(port_data[i][1]);
		    count++;
		}
	}
			
	// zero_idx = (CF_pro(:,:,FloBonds)==0)
			
    double*** zero_idx = new3Darray(Max_tenor, Frac_Year + 1, Flo_count); 
    for(int i = 0; i < Max_tenor;i++){
        for(int j = 0; j < Frac_Year + 1;j++){
            int count = 0;
            for(int k = 0; k < port_row_number;k++){
            	if(FloBonds[k] == 1){
				    if(CF_pro[i][j][k] == 0)
				        zero_idx[i][j][count] = 1;
				    count++;
				}
			}
		}
	}
	// flo_cpnrate = future_forswp(:,:,flobounds_cur)
	double*** flo_cpnrate = new3Darray(Max_tenor, Frac_Year + 1, Flo_count);
	for(int i = 0; i < Max_tenor;i++){
		for(int j = 0; j < Frac_Year + 1;j++){
		    for(int k = 0; k < Flo_count;k++){
		        if(zero_idx[i][j][k] == 1)
		        	flo_cpnrate[i][j][k] = 0;
		        else
			        flo_cpnrate[i][j][k] = future_forswp_reshape[i][j][flobonds_cur[k] - 1];
			}
		}
	}
	double* flo_RempP = new double[Flo_count]();
	count = 0;
	for(int i = 0; i < port_row_number;i++){
	    if(FloBonds[i] == 1){
	        flo_RempP[count] = port_data[i][13];
	        count++;
		}
	}
	int* FloBonds_idx = new int[Flo_count]();
	count = 0;
	for(int i = 0; i < port_row_number;i++){ 
        if(FloBonds[i] == 1){		    
			FloBonds_idx[count] = i + 1;
			count++;
		}
	}
	double*** end_cfidx = new3Darray(Max_tenor, Frac_Year + 1, Flo_count);
	for(int i = 0; i < Flo_count;i++){
		for(int j = 0; j < Max_tenor;j++){
			for(int k = 0; k < Frac_Year + 1;k++){
				if(CF_pro[j][k][FloBonds_idx[i]-1] > flo_RempP[i])
				    end_cfidx[j][k][i] = 1;					
			}
		}
    }  
    for(int i = 0; i < Flo_count;i++){
        double*** idx = new3Darray(Max_tenor, Frac_Year + 1, Flo_count);
		for(int j = 0; j < Max_tenor;j++){
			for(int k = 0; k < Frac_Year + 1;k++){
				for(int l = 0; l < Flo_count;l++){
					if(l == i)
						idx[j][k][l] = end_cfidx[j][k][i]; 																		
				}
			}
		}
		for(int j = 0; j < Max_tenor;j++){
			for(int k = 0; k < Frac_Year + 1;k++){
				for(int l = 0; l < Flo_count;l++){
					if(idx[j][k][l] ==  1)
						flo_cpnrate[j][k][l] = flo_cpnrate[j][k][l] + flo_RempP[i]; 																		
				}
			}
		}
    }
	for(int j = 0; j < Max_tenor;j++){
		for(int k = 0; k < Frac_Year + 1;k++){
			int count = 0;
			for(int l = 0; l < Flo_count;l++){
				if(FloBonds[l] ==  1){
					CF_pro[j][k][l] = flo_cpnrate[j][k][count]; 
					count++;	
				}
			}
	    }
	}       
   
    double* IP = new double[port_row_number]();
    im_spread = new double[port_row_number]();
    column_index = indexing(Portfolio_Table.file[0], "Spread");
    for(int i = 0; i < port_row_number;i++){
        IP[i] = port_data[i][8];
        im_spread[i] = stod(Portfolio_Table.file[i + start_row][column_index])/10000;
	}
    cout << "Pricing Module." << endl;
    cout << "Calibration & Projection..." << endl;
    switch(Pricing_method){
    	case 1 : {
    		cout << "Pricing with Single Path Method." << endl;
    		for(int i = 0; i < port_row_number;i++){  
    			if(cal_idx[i] == 1){
				    // Prepare for the input of function CF_Discount_Price_SP
				    double* CF_m = new double[period];
    			    double** CF = new2Darray(Max_tenor, period);    
    			    double** Forward = new2Darray(Max_tenor, period);
    			    double* CF_factor = new double[period];
				    double* AI = new double[period]; 
				    double** callable_t = new2Darray(Max_tenor, period);
				    double* call_flag = new double[period];
    			    double* nominfreq = new double[period];
    			    double* Spread = new double[period];
    			    for(int j = 0; j < period;j++){
    				    CF_m[j] = CF_end_time_pro[i][j];
    				    CF_factor[j] = cfac_pro[i][j];
    				    AI[j] = AI_pro[i][j];
    				    call_flag[j] = call_f_pro[i][j];
    				    nominfreq[j] = comf_pro[i][j];
    				    Spread[j] = im_spread[i];
    				    for(int k = 0; k < Max_tenor;k++){
    				    	CF[k][j] = CF_pro[k][j][i];
						    Forward[k][j] = Forward_Curve[k][j][i];
						    callable_t[k][j] = callT_pro[k][j][i]; 
						}
				    }
				    // Start function
				    CF_DISCOUNT result = CF_Discount_Price_SP(CF_m, CF, Forward, CF_factor, AI, callable_t, call_flag, nominfreq, Frequency, Spread);
                    for(int j = 0; j < period;j++){
                        Clean_Price[i][j] = result.Clean_Price[j];
                        Dirty_Price[i][j] = result.Dirty_Price[j];					
					}
				}
			}	
			break;
		}
    	case 2 :{
    		// Not Done yet
			break;
		}   	
	}
	
    column_index = indexing(Portfolio_Table.file[0], "IMspread");
    for(int i = 0; i < port_row_number;i++){
        port_data[i][21] = im_spread[i];
        Portfolio_Table.file[i][column_index] = to_string(im_spread[i]);
        for(int j = 0; j < period;j++){
        	port_data[i][22+j] = Clean_Price[i][j];
		}
	}
	// End Cash Flow module //
	////////////////////////////////////////////////////////////
	// Amortization Module //
	cout << "Amortization Module Begins." << endl;
	double** amort_table = new2Darray(port_row_number, 17);
	double pricing_num = datenum(Pricing_Y, Pricing_M, Pricing_D);

	// 1. Bond type
	column_index = indexing(Portfolio_Table.file[0], "BondType");
	for(int i = 0; i < port_row_number; i++){	    
		amort_table[i][0] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}

	// 2. AmortizationCost 
	column_index = indexing(Portfolio_Table.file[0], "AmortizedCost");
	for(int i = 0; i < port_row_number; i++){	    
		amort_table[i][1] = stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	// 3. Maturitynum
	for(int i = 0; i < port_row_number; i++){
		int MY = stoi(Portfolio_Table.file[i + start_row][column_index_my]);
		int MM = stoi(Portfolio_Table.file[i + start_row][column_index_mm]);
		int MD = stoi(Portfolio_Table.file[i + start_row][column_index_md]);
		amort_table[i][2] = datenum(MY, MM, MD);
	}	
	
	// 4. Callnum
	for(int i = 0; i < port_row_number; i++){
		int CY = stod(Portfolio_Table.file[i + start_row][column_index_cy]);
		int CM = stod(Portfolio_Table.file[i + start_row][column_index_cm]);
		int CD = stod(Portfolio_Table.file[i + start_row][column_index_cd]);
		amort_table[i][3] = datenum(CY, CM, CD);
	}
	
	// 5. CouponFrequency
	column_index = indexing(Portfolio_Table.file[0], "CouponFrequency");
	for(int i = 0; i < port_row_number; i++){
		if(stod(Portfolio_Table.file[i + start_row][column_index]) == 0)
		    amort_table[i][4] = 0;    
	    else
		    amort_table[i][4] = 12 / stod(Portfolio_Table.file[i + start_row][column_index]);
	}
	
	// 6. CouponRate
	column_index = indexing(Portfolio_Table.file[0], "Coupon");
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][5] = stod(Portfolio_Table.file[i + start_row][column_index]);
    }
    // 7. Redemption
	column_index = indexing(Portfolio_Table.file[0], "Redemption");
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][6] = stod(Portfolio_Table.file[i + start_row][column_index]);
    }
    
    // 8. BookYield
	column_index = indexing(Portfolio_Table.file[0], "Yield-to-MaturityAtIssue");
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][7] = stod(Portfolio_Table.file[i + start_row][column_index]);
    }

	// 9. CallFlag
	column_index = indexing(Portfolio_Table.file[0], "CallFlag");
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][8] = stod(Portfolio_Table.file[i + start_row][column_index]);
    }
    
    // 10. amortnum
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][9] = amort_table[i][2];
    }
    
    for(int i = 0; i < port_row_number; i++){
    	if(amort_table[i][8] != 0 && amort_table[i][8] != 2)
	    amort_table[i][9] = amort_table[i][3];
    }

    // 11. FloSpread
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][10] = port_data[i][11];
    } 
    
    // 12. curidx
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][11] = port_data[i][1];
    }
    
    // 13. floidx
	for(int i = 0; i < port_row_number; i++){
	    amort_table[i][12] = port_data[i][9];
    }
    
    // 14. Amort_idx
    int cal_idx_num = 0;
	for(int i = 0; i < port_row_number; i++){
	    if(((((amort_table[i][1] == 0 || amort_table[i][1] != amort_table[i][6]) && (amort_table[i][0] != 0))
	        && (amort_table[i][4] != 0)) && (amort_table[i][5] != 0))){
	        cal_idx[i] = 1;
			cal_idx_num++;
		}
	    amort_table[i][13] = cal_idx[i];
    }    
    //AM_cfdates
    
	double* amort_table_mat_cal = new double[cal_idx_num]();
	double* amort_table_cou_cal = new double[cal_idx_num]();	    
    count = 0;
	for(int i = 0; i < port_row_number; i++){
	    if(cal_idx[i] == 1){
	        amort_table_mat_cal[count] = amort_table[i][2];
	        amort_table_cou_cal[count] = amort_table[i][4];
            count++;
	    }
	}
    cout << "test";
	double** CpDates_tmp =  AM_cfdates(pricing_num, amort_table_mat_cal, amort_table_cou_cal);
    double** CouponDates = new2Darray(port_row_number, arraysize(CpDates_tmp, 2));
	for(int i = 0; i < port_row_number; i++){
		if(cal_idx[i] == 1){
	        for(int j = 0; j < arraysize(CpDates_tmp, 2);j++)
	            CouponDates[i][j] = CpDates_tmp[i][j];			        
		}
		else{
			for(int j = 0; j < arraysize(CpDates_tmp, 2);j++)
	            CouponDates[i][j] = 999;
		}
	} 
	
	for(int i = 0; i < port_row_number; i++){
		if(amort_table[i][0] == 2)
	        amort_table[i][9] = CouponDates[i][1];
    }
	
	// 浮動債債券攤銷到最近重設日
	cout << "Calculating amortized IRR." << endl; 
	
	double* IRR = new double[cal_idx_num]();
	count = 0;
	for(int i = 0; i < port_row_number; i++){
	    if(cal_idx[i] == 1){
	        IRR[count] = AM_CaliamortIRR(pricing_num, amort_table[i][2], amort_table[i][9], amort_table[i][6],
			             amort_table[i][5], amort_table[i][4], amort_table[i][1], amort_table[i][7]);
            count++;
	    }
	}

    // 15. amort_IRR
    count = 0;
    for(int i = 0; i < port_row_number; i++){
	    if(cal_idx[i] == 1){
	        amort_table[i][14] = IRR[count];
	        count++;
	    }else{
	    	amort_table[i][14] = 0;
		}
	}
	
	cout << "Constructing Daily Amortization." << endl;
	double** AI_factor = new2Darray(port_row_number, arraysize(CpDates_tmp, 2) - 1);
    for(int i = 0; i < port_row_number; i++){
	    if(cal_idx[i] == 1){
	        for(int j = 0; j < arraysize(CpDates_tmp, 2) - 2; j++){
                AI_factor[i][j+1] = AM_AIfactor(CouponDates[i][j], CouponDates[i][j+1], CouponDates[i][j+1]);
	        }
        }else{
        	for(int j = 0; j < arraysize(CpDates_tmp, 2) - 1; j++){
                AI_factor[i][j] = 999;
	        }
		}
	}

	double** CashFlow = new2Darray(port_row_number, arraysize(CpDates_tmp, 2) - 1);
	for(int i = 0; i < port_row_number;i++){
        for(int j = 0; j < arraysize(CpDates_tmp, 2) - 1;j++){
        	if(AI_factor[i][j] != 999 && amort_table[i][5] != 999 && amort_table[i][6] != 999){
        	    CashFlow[i][j] = AI_factor[i][j]*amort_table[i][5]*amort_table[i][6];
        	}else{
        		CashFlow[i][j] = 999;
			}     	
		}
	}
	
	for(int i = 0; i < port_row_number;i++){
		int record = arraysize(CpDates_tmp, 2) - 2;
        for(int j = 0; j < arraysize(CpDates_tmp, 2) - 1;j++){
        	if(CashFlow[i][j] == 999){
        	    record = j - 1;
        	  	break;
        	}
		}
		CashFlow[i][record] += amort_table[i][6];
	}
	
	int EndProject_M = ((int(Pricing_M + 12 / Projected_year_Frequency - 1) % 12) + 12) % 12 + 1;
	int EndProject_Y = Pricing_Y + floor((Pricing_M + 12 / Projected_year_Frequency - 1)/12);
	int EndProject_D = Pricing_D;
	if(Pricing_D == 31||(Pricing_D == 28 && EndProject_M == 2))
	    EndProject_D = 30;
	double end_date = datenum(EndProject_Y, EndProject_M, EndProject_D);
	double** Ref_Date = new2Darray(port_row_number, arraysize(CpDates_tmp, 2));
	for(int i = 0; i < port_row_number;i++){
        for(int j = 0; j < arraysize(CpDates_tmp, 2);j++){
        	if(CouponDates[i][j] != 999){
        	    if(CouponDates[i][j] > end_date)
        	        Ref_Date[i][j] = CouponDates[i][j];
        	    else
        	        Ref_Date[i][j] = 999;
        	}else{
        		Ref_Date[i][j] = 999;
			}
		}	
	}
	
	vector<double> tempvec1;
	for(int i = 0; i < port_row_number;i++){
		vector<double> tempvec2;
	    for(int j = 0; j < arraysize(CpDates_tmp, 2);j++){
	        if(Ref_Date[i][j] != 999)
	            tempvec2.push_back(Ref_Date[i][j]);
		}
	    tempvec1.push_back(*min_element(tempvec2.begin(), tempvec2.end()));
	}
	double End_Date = *max_element(tempvec1.begin(), tempvec1.end());
    
    double* iter_date = new double[int(End_Date) - int(pricing_num) + 2]();
    for(int i = 0; i < int(End_Date) - int(pricing_num) + 2;i++)
        iter_date[i] = i + pricing_num - 1;  
    
	// For Coupon Rates of Floating Bonds
	double*** Intrp_SptCur = new3Darray(Max_tenor, int(End_Date - pricing_num), currency_number);
	for(int i = 0; i < Max_tenor;i++){
		for(int j = 0; j < currency_number;j++){
			// First create the matrix as input of the interpolation function
			double grid = double(Frequency)/double(Projected_year_Frequency*(int(End_Date - pricing_num) - 1));
			for(int k = 0; k < int(End_Date - pricing_num) - 1;k++){
				double point = 1 + grid * k;
			    double start = future_forswp_reshape[i][int(point)-1][j];
			    double end = future_forswp_reshape[i][int(point)][j];
			    double value = (end - start)*(point - int(point)) + start;
			    Intrp_SptCur[i][k][j] = value;
		    }
		    Intrp_SptCur[i][int(End_Date - pricing_num) - 1][j] = future_forswp_reshape[i][int(Frequency/Projected_year_Frequency)][j];
		}
	}

    double** cpndates = new2Darray(port_row_number, arraysize(CpDates_tmp, 2));
    for(int i = 0; i < port_row_number;i++){
        for(int j = 0; j < arraysize(CpDates_tmp, 2);j++){
        	if(CouponDates[i][j] != 999)
                cpndates[i][j] = CouponDates[i][j] - CouponDates[i][0];	
			else
			    cpndates[i][j] = 999;	
		}
    }
    
    int num_flobonds;
    idx = new int[port_row_number]();
    for(int i = 0; i < port_row_number;i++){
        if(amort_table[i][0] == 2 && cal_idx[i] == 1){
            num_flobonds++;
            idx[i] = 1;
        }
    }
    
    //// Delete it after using ////
	idx[0] = 1;
	idx[1] = 1;
	idx[2] = 1;
	idx[3] = 1;
	num_flobonds = 4;
 	 
    double** FloCpnRate = new2Darray(port_row_number, arraysize(CpDates_tmp, 2));
    for(int i = 0; i < port_row_number;i++){
    	if(idx[i] == 1){
            vector<int> reset_point;
            for(int j = 0; j < arraysize(CpDates_tmp, 2);j++){
        	    if(cpndates[i][j] < int(End_Date - pricing_num) && cpndates[i][j] != 999)
        	        reset_point.push_back(j);
        	    FloCpnRate[i][j] = amort_table[i][5];    
		    }
		    if(reset_point.size() > 2){
			    int Cur = int(amort_table[i][12]);
			    int Tenor = 12/int(amort_table[i][4]) + 1;
			    vector<double> CpnR_tmp;
			    for(int k = 2; k < reset_point.size();k++){
				    if(Intrp_SptCur[Tenor][reset_point[k]][Cur] != 999 && amort_table[i][10] != 999)
				        CpnR_tmp.push_back(Intrp_SptCur[Tenor][reset_point[k]][Cur] + amort_table[i][10]);
				    else
				        CpnR_tmp.push_back(999);
			    }
				for(int m = 2; m < reset_point.size();m++)
				    FloCpnRate[i][reset_point[m]] = CpnR_tmp[m];
				for(int m = reset_point.back(); m < arraysize(CpDates_tmp, 2);m++)    
				    FloCpnRate[i][m] = CpnR_tmp.back();
		    }
	    }
    }
    
    double*** amort_info = AM_Amortization(amort_table, pricing_num, End_Date, CouponDates, FloCpnRate);
    
	int amort_info_number = _msize(amort_info)/sizeof(double);
  
    //Scientific Amortization
    
    double** idx_zeroAI = new2Darray(amort_info_number, port_row_number);
    for(int i = 0; i < amort_info_number;i++){
    	if(i == 0 || i == amort_info_number - 1){
            for(int j = 0; j < port_row_number;j++)
        	    idx_zeroAI[i][j] = 1;            
        }
		else{
            for(int j = 0; j < port_row_number;j++){
        	    if(amort_info[i][1][j] == 0)
        	        idx_zeroAI[i][j] = 1;
            }
        }
    }
    
    double** BV_Clean = new2Darray(amort_info_number, port_row_number);
    for(int i = 0; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
        	    BV_Clean[i][j] = amort_info[i][4][j];
        }
    }
    
    double** BC_Scientific = new2Darray(port_row_number, amort_info_number);
    for(int i = 0; i < port_row_number;i++){
    	double** tmp = new2Darray(amort_info_number, 2);
    	for(int j = 0; j < amort_info_number;j++){
    		tmp[j][0] = iter_date[j];
    		if(idx_zeroAI[j][i] == 1)
    		    tmp[j][1] = BV_Clean[j][i];
    		else
    		    tmp[j][1] = 999;
		} 
    	
		double** result = interpolation(tmp, amort_info_number, 2, "linear");
    	for(int j = 0; j < amort_info_number;j++){
    	    BC_Scientific[i][j] = result[j][1];    
		}    	
	}

    double*** amort_info_final = new3Darray(amort_info_number, 20, port_row_number);
    for(int i = 0; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
        	for(int k = 0; k < 6;k++){
        		amort_info_final[i][k][j] = amort_info[i][k][j];
			}
        	amort_info_final[i][6][j] = BC_Scientific[j][i];      	
        }
    }
    
    for(int i = 1; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
            amort_info_final[i][7][j] = amort_info_final[i][2][j] + amort_info_final[i][5][j] - 
			                            amort_info_final[i-1][2][j] - amort_info_final[i-1][5][j];
	    }
    }
    
    double* FX_interped = new double[amort_info_number];
    for(int i = 0; i < amort_info_number;i++){
    	FX_interped[i] = FX_USD[0] + (FX_USD[1] - FX_USD[0])*(iter_date[i] - Date_Data[0])/(Date_Data[1] - Date_Data[0]);  	
	}

	for(int i = 0; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
            amort_info_final[i][14][j] = FX_interped[i];
	    }
    }
 
    for(int i = 0; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
            amort_info_final[i][15][j] = Clean_Price[j][0] + (Clean_Price[j][1] - Clean_Price[j][0])*(iter_date[i] - Date_Data[0])/(Date_Data[1] - Date_Data[0]);
	    }
    }

    for(int i = 1; i < amort_info_number;i++){
        for(int j = 0; j < port_row_number;j++){
            amort_info_final[i][8][j] = (amort_info_final[i][14][j] - amort_info_final[i-1][14][j]) * amort_info_final[i-1][2][j];
            amort_info_final[i][9][j] = (amort_info_final[i][14][j] - amort_info_final[i-1][14][j]) * amort_info_final[i-1][5][j];
            amort_info_final[i][10][j] = (amort_info_final[i][14][j] - amort_info_final[i-1][14][j]) * amort_info_final[i-1][4][j];
	        amort_info_final[i][11][j] = (amort_info_final[i][14][j]*amort_info_final[i][15][j]) - (amort_info_final[i-1][14][j]*amort_info_final[i-1][15][j]);
		}
    }
    

    print3Darray(amort_info_final,-1,15,-1);
	return 0; 
	
}
